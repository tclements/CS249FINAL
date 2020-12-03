using CSV, DataFrames, DSP, HDF5, Random, Statistics

function fp32int8(A::AbstractArray)
    A .-= mean(A,dims=1)
    maxs = maximum(abs,A,dims=1)
    factors = 127 ./ maxs
    A .*= factors
    return round.(Int8,A)
end 

function fp32uint8(A::AbstractArray)
    m = min(minimum(A),0)
    factor = 255 / (maximum(A) - m)
    return round.(UInt8,(A .- m) .* factor)
end

function vel2acc(A::AbstractArray,freqs::AbstractArray)
    Xw = fft(A,1)
    Xw .*= freqs .* 1im .* 2π
    return real.(ifft(Xw,1))
end

function taper!(A::AbstractArray;α::Real=0.05)
    window = tukey(size(A,1),α)
    A .*= window
    return nothing
end

function randamp!(A::AbstractArray)

    return nothing
end

function wav2spec(A::AbstractArray,n,overlap,fs)
    spec = spectrogram(A[:,1],n,overlap,fs=fs)
    Nrows, Ncols = size(spec.power)
    out = zeros(eltype(A),Nrows,Ncols,3)
    out[:,:,1] .= spec.power
    for ii = 2:3
        out[:,:,ii] .= spectrogram(A[:,ii],n,overlap,fs=fs).power
    end
    
    # make smallest value across channels equal to 1 
    out .+= (1 - minimum(out))\

    # return log10 of power 
    return log10.(out)
end

function ricker(f,n,dt,t0)
    # Create the wavelet and shift in time if needed
    T = dt*(n-1)
    t = 0:dt:T
    tau = t .- t0
    s = (1 .-tau .* tau .* f^2 * pi^2) .* exp.(-tau .^ 2 .* pi^2 .*f^2)
    return s
end

function specplot(A::AbstractArray;n=50,overlap=45,fs=100.)
    spec = spectrogram(A,n,overlap,fs=fs)
    t = 0 : 1 / fs : size(A,1) / fs - 1 / fs
    p1 = plot(t,A)
    xlims!(p1,minimum(spec.time),maximum(spec.time))
    p2 = heatmap(
        spec.time,
        spec.freq,
        10 .* log10.(spec.power),
        legend=:none,
        xlabel="Time [s]",
        ylabel="Frequency [Hz]",
    )
    l = @layout [a;b]
    p = plot(p1,p2, layout = l)
    display(p)
    return nothing 
end

function highpass!(A::AbstractArray{<:AbstractFloat},freq::Real,fs::Real)
    T = eltype(A)
    # create filter
    responsetype = Highpass(T(freq); fs=fs)
    designmethod = Butterworth(T,4)
    A[:,:] .= filt(digitalfilter(responsetype, designmethod), @view(A[:,:]))
    return nothing
end

function detrend!(X::AbstractArray{<:AbstractFloat})
    T = eltype(X)
    N = size(X,1)

    # create linear trend matrix
    A = similar(X,T,N,2)
    A[:,2] .= T(1)
    A[:,1] .= range(T(0),T(1),length=N)
    # create linear trend matrix
    R = transpose(A) * A

    # do the matrix inverse for 2x2 matrix
    # this is really slow on GPU
    Rinv = inv(R)
    factor = Rinv * transpose(A)

    # remove trend
    X .-= A * (factor * X)
    return nothing
end

function builddirs(DATADIR)
    WAVES = joinpath(DATADIR,"waves")
    WAVETRAIN = joinpath(WAVES,"train")
    WAVETEST = joinpath(WAVES,"test")
    SPECS = joinpath(DATADIR,"specs")
    SPECTRAIN = joinpath(SPECS,"train")
    SPECTEST = joinpath(SPECS,"test")
    for dir in [WAVES,WAVETRAIN,WAVETEST,SPECS,SPECTRAIN,SPECTEST]
        if !isdir(dir)
            mkpath(dir)
        end
    end

    # make directories for each class 
    for dir in [WAVETRAIN,WAVETEST,SPECTRAIN,SPECTEST]
        for class in ["noise","p-wave","s-wave"]
            tmpdir = joinpath(dir,class)
            if !isdir(tmpdir)
                mkpath(tmpdir)
            end
        end
    end
    return WAVES, WAVETRAIN, WAVETEST, SPECS, SPECTRAIN, SPECTEST
end

function test_train_split(df::DataFrame)
    ind = findall(df[:,:receiver_type] .∈ Ref(["HH","EH"]))
    Hdf = df[ind,:]

    # split into test and train datasets 
    trainind = findall((Hdf[:,:receiver_latitude] .> 30) .& (Hdf[:,:receiver_latitude] .< 50) .& (Hdf[:,:receiver_longitude] .> -130) .& (Hdf[:,:receiver_longitude] .< -108))
    testind = collect(1:size(Hdf,1))
    deleteat!(testind,trainind)

    # filter out data in W Canada and Texas
    CDind = findall((Hdf[:,:receiver_latitude] .> 50) .& (Hdf[:,:receiver_latitude] .< 55.5) .& (Hdf[:,:receiver_longitude] .> -130.5) .& (Hdf[:,:receiver_longitude] .< -112))
    TXind = findall((Hdf[:,:receiver_latitude] .> 30) .& (Hdf[:,:receiver_latitude] .< 50) .& (Hdf[:,:receiver_longitude] .> -109) .& (Hdf[:,:receiver_longitude] .< -101.5))
    filter!(x-> x ∉ CDind,testind)
    filter!(x-> x ∉ TXind,testind)

    # separate into NOISE and EQ datasets for test / train 
    EQtrainind = findall(.!ismissing.(Hdf[trainind,:p_status]))
    EQtestind = findall(.!ismissing.(Hdf[testind,:p_status]))
    noisetrainind = findall(ismissing.(Hdf[trainind,:p_status]))
    noisetestind = findall(ismissing.(Hdf[testind,:p_status]))
    EQtrain = Hdf[trainind[EQtrainind],:]
    EQtest = Hdf[testind[EQtestind],:]
    NOISEtrain = Hdf[trainind[noisetrainind],:]
    NOISEtest = Hdf[testind[noisetestind],:]

    # take 80 / 20 split for test / train EQ 
    NEQtest = size(EQtest,1)
    NEQtrain = NEQtest * 4 
    train80 = randperm(size(EQtrain,1))[1:NEQtrain] 
    EQtrain = EQtrain[train80,:]
    EQtest = EQtest[randperm(NEQtest)[1:NEQtest],:]

    # even out number of stations for NOISE datasets 
    NNtest = size(NOISEtest,1)
    NNtrain = size(NOISEtrain,1)
    NN = min(NNtest,NNtrain)
    NOISEtest = NOISEtest[randperm(NNtest)[1:NN],:]
    NOISEtrain = NOISEtrain[randperm(NNtrain)[1:NN],:]
    return EQtrain, EQtest, NOISEtrain, NOISEtest
end

function augment_noise!(A::AbstractArray,spikes,ramp,)
end


function preprocess_noise(
    df::DataFrame,
    datafile,
    DATADIR,
    nwindows;
    train=true,
    α=0.065,
    freq=2.,
    window_length=2.,
    n = 50,
    overlap=45,
    fs=100.,
)

    # load (meta)data
    h5file = h5open(datafile,"r")
    N = size(df,1)
    if train
        SPEC = joinpath(DATADIR,"specs","train","noise")
        WAVE = joinpath(DATADIR,"waves","train","noise")
    else
        SPEC = joinpath(DATADIR,"specs","test","noise")
        WAVE = joinpath(DATADIR,"waves","test","noise")
    end

    # start after taper 
    window_samples = round(Int,window_length * 100) 
    total_windows = round(Int,6000 / window_samples)
    startwindow = round(Int,6000 * α / window_samples + 1)
    for ii = 1:N
        path = df[ii,:trace_name]
        println("Reading file $path $ii of $N")
        data = read(h5file,"data/" * path) |> transpose |> Array
        detrend!(data)
        taper!(data,α=α)
        highpass!(data,freq,100.)
        for ii = startwindow:min(total_windows - startwindow,startwindow+nwindows-1)
            wave = data[ii*window_samples:(ii+1)*window_samples-1,:]
            qwave = fp32uint8(wave)
            spec = wav2spec(wave,n,overlap,fs)
            qspec = fp32uint8(spec)
            waveout = joinpath(WAVE,"$(path)_$ii.jpg")
            specout = joinpath(SPEC,"$(path)_$ii.jpg")
            save(waveout,qwave)
            save(specout,qspec)
        end
    end
    close(h5file)
    return nothing
end

function preprocess_waves(df::DataFrame,datafile,DIR)
    # load (meta)data
    h5file = h5open(datafile,"r")
    N = size(df,1)
    PWAVE = joinpath(DIR,"pwave")
    SWAVE = joinpath(DIR,"swave")
    for ii = 1:N
        path = df[ii,:trace_name]
        data = read(file,"data/" * path) |> transpose |> Array
        detrend!(data)
        taper!(data,α=0.1)
        highpass!(data,2.,100.)
    end
end

### training workflow 
# 1. read from HDF5
# 2. detrend
# 3. taper
# 4. highpass filter above 1 Hz 
# 4. convert to acceleration 
# 5. window earthquakes around arrival randomly
# 5a. add gaussian normal noise 
# 5b. clip amplitudes randomly 
# 6. convert to spectrogram or not 
# 6a. perform spectral averaging  
# 7. quantization 
# 8. save into train / test 

# directory structure 
BASEDIR = expanduser("~/CS249FINAL")
DATADIR = joinpath(BASEDIR,"data")
WAVES, WAVETRAIN, WAVETEST, SPECS, SPECTRAIN, SPECTEST = builddirs(DATADIR)

# load metadata 
df = CSV.File(joinpath(DATADIR,"merge.csv"),"r") |> DataFrame
EQtrain, EQtest, NOISEtrain, NOISEtest = test_train_split(df)

# need 22 two-second windows per training noise sample 
# need 6 two-second windows per test noise sample 


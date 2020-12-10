import glob, os 
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np 
import matplotlib.pyplot as plt 

def test2input(A,input_dir):
    'Convert test data to cc/header file'

    # files to write 
    cc_file = os.path.join(input_dir,"input_data.cc")
    h_file = os.path.join(input_dir,"input_data.h")

    # create header file first 
    h_out = "#ifndef EDE_INPUT_DATA_H_\n" \
            "#define EDE_INPUT_DATA_H_\n\n" \
            "extern const float input_data[];\n" \
            "#endif\n"
    open(h_file, "w").write(h_out)

    # write data to cc file 
    A = A.flatten(order="F")
    cc_out = '#include "input_data.h"\n' \
             "static const int input_data_len = 240;\n" \
             "static const float input_data[240] = {\n"
    arrstring = ""
    for ii in range(A.size-1):
        arrstring += str(A[ii])
        arrstring += ", "
    arrstring += str(A[-1])
    arrstring += "};\n"
    cc_out += arrstring
    open(cc_file, "w").write(cc_out)
    return None

if __name__ == "__main__": 

    # load test data 
    testdata = np.load("/home/timclements/CS249FINAL/data/test.npz")
    Xtest = testdata["Xtest"]
    Ytest = testdata["Ytest"]

    # check that models work on test data 
    models = glob.glob("/home/timclements/CS249FINAL/models/*")
    model = tf.lite.Interpreter(models[0])
    model.allocate_tensors()
    model_input_index = model.get_input_details()[0]["index"]
    model_output_index = model.get_output_details()[0]["index"]
    x_value_tensor = tf.convert_to_tensor(Xtest[0:1,:,:,:], dtype=np.float32)
    model.set_tensor(model_input_index, x_value_tensor)
    model.invoke()
    model.get_tensor(model_output_index)[0]

    # convert some test data to data that can be read on the device 
    input_dir = "/home/timclements/CS249FINAL/src/"
    test2input(Xtest[0:1],input_dir)




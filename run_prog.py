from scipy import io
import fully_connected_auto
import matplotlib.pyplot as plt
for index in range(17,18):
    string='data_save'
    a,b,c,d,e=fully_connected_auto.ggg_auto(1)
    string +=str(index)
    data={'pred_error':a,'W1':b,'W2':c,'b1':d,'b2':e}
    io.savemat(string,data)
    #plt.plot(a)
#plt.plot(y1)
    #plt.show()
    

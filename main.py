import random
import math
import matplotlib.pyplot as plt

class neural_network:
    def __init__(self,inp,h,o):#initialise network
        self.inputs=[1 for _ in range(inp)]
        self.hidden=[[[1,0] for _ in range(h[i])] for i in range(len(h))]#neuron value,bias
        self.outputs=[[1,0] for _ in range(o) ]
        self.weights=[]

        self.weights.append([[random.uniform(-1,1) for i in range(h[0])] for _ in range(inp)])

        for i in range(len(h)-1):
            self.weights.append([[random.uniform(-1,1) for n in range(h[i+1])] for w in range(h[i])])

        self.weights.append([[random.uniform(-1,1) for i in range(o)] for _ in range(h[-1])]) 
        #layer indexing is [layer][source neuron][target neuron]
       
    def sigmoid(self,n):#sigmoid activation
        return 1/(1+2.718**(-n))
    def sigmoid_derivative(self,n):
        return self.sigmoid(n)*(1-self.sigmoid(n))
    def mse(self,predval,realval):#mean squared error
        return (realval-predval)**2
    def mse_derivative(self,predval,realval):
        return 2*(predval-realval)
   
    def forward(self,inputvalues):#forward propagation- calculates outputs
        if len(inputvalues)!=len(self.inputs):
            raise ValueError("input size doesnt match input network size")
        self.inputs=inputvalues
        for h in range(len(self.hidden[0])):#iterating through neurons in hidden layer 0
            nsum=0#sum of all input neurons * weight connected to it
            for n in range(len(self.inputs)):#iterating through input neurons
                nsum+=self.inputs[n]*self.weights[0][n][h]
            nsum=self.sigmoid(nsum+self.hidden[0][h][1])#adds the bias then applies sigmoid activation
            self.hidden[0][h][0]=nsum
        if len(self.hidden)!=1:
            for layer in range(len(self.hidden)-1):
                for h in range(len(self.hidden[layer+1])):
                    nsum=0
                    for n in range(len(self.hidden[layer])):
                        nsum+=self.hidden[layer][n][0]*self.weights[layer+1][n][h]
                    nsum=self.sigmoid(nsum+self.hidden[layer+1][h][1])
                    self.hidden[layer+1][h][0]=nsum
        output_values=[0 for i in range(len(self.outputs))]
        for o in range(len(self.outputs)):#calculate output layer
            nsum=0
            for n in range(len(self.hidden[-1])):
                nsum+=self.hidden[-1][n][0]*self.weights[-1][n][o]
            nsum=nsum+self.outputs[o][1]
            self.outputs[o][0]=nsum
            output_values[o]=nsum
        return output_values
    
    def backward(self,target_values):#backpropagation one pass
        learning_rate=0.1#how quickly it converges
        inputs=self.inputs
        pred_values=self.forward(inputs)
        values_length=len(pred_values)
        if len(target_values)!=values_length:
            raise ValueError("target value size doesnt match predicted values size")
        errors=[self.mse(pred_values[i],target_values[i]) for i in range(values_length)]
        def calculategradients():
            #this is only temporary for last layer only will change soon
            error_derivatives=[self.mse_derivative(pred_values[i],target_values[i]) for i in range(values_length)]
           
            #partial derivative with respect to weights in last layer
            llayer_dx=[[0 for _ in range(len(self.outputs))]for _ in range(len(self.hidden[-1]))]
            
            #calculate partial derivatives for weights in last layer
            for i in range(len(self.hidden[-1])):
                for j in range(len(self.outputs)):
                    #indexing [hiddenneuron][output]
                    llayer_dx[i][j]=error_derivatives[j]*self.hidden[-1][i][0]
            #calculate partial derivatives for weights in hidden layers
            lweight_dx = [[[0 for _ in range(len(self.hidden[l]))]\
               for _ in range(len(self.inputs if l == 0 else self.hidden[l-1]))] for l in range(len(self.hidden))]
            lbias_dx=[[0 for _ in range(len(self.hidden[l]))]for l in range(len(self.hidden))]
            #print(lweight_dx)
            #seems innacurate
            for layer in range(0,len(self.hidden)):
                targetlayer=len(self.hidden)-layer-1
                tlayer=self.hidden[targetlayer]
                inputlayer=self.hidden[targetlayer-1] if targetlayer!=0 else self.inputs
                newerror_derivatives=[0 for _ in range(len(self.hidden[targetlayer]))]
                for tar in range(len(self.hidden[targetlayer])):
                    newerror_derivatives[tar]=sum(error_derivatives[i]*self.weights[targetlayer+1][tar][i]\
                       for i in range(len(error_derivatives)))
                    targetval=self.hidden[targetlayer][tar][0]
                    grad=targetval*(1-targetval)*newerror_derivatives[tar]
                    biasgrad=grad
                    lbias_dx[targetlayer][tar]=biasgrad
                    for inp in range(len(inputlayer)):
                        sourceval=inputlayer[inp] if targetlayer==0 else inputlayer[inp][0]
                        weightgrad=grad*sourceval
                        #print(inp)
                        #print(inputlayer)
                        
                        lweight_dx[targetlayer][inp][tar]=weightgrad
                error_derivatives=newerror_derivatives
                
            print(lweight_dx)

 
            return llayer_dx,error_derivatives,lweight_dx,lbias_dx
        gradl,gradbl,gradwh,gradbh=calculategradients()
        #perform one backwards pass on every weight in the last layer ONLY TEMPORARY for debugging
        for i in range(len(self.hidden[-1])):
            for j in range(len(self.outputs)):
                self.weights[-1][i][j]-=gradl[i][j]*learning_rate
        for o in range(len(self.outputs)):self.outputs[o][1]-=gradbl[o]*learning_rate
        self.weights[0][0][0]-=gradwh[0][0][0]*learning_rate
        self.hidden[0][0][1]-=gradbh[0][0]*learning_rate#temporary for debugging
            
       
    
   
#____interface- not related to neuralnetworkclass___________
nn=neural_network(1,[5,5],1)
inputvals=[i for i in range(1,11)]
targetvals=[math.sqrt(i) for i in range(1,11)]
predictedvals=[0 for _ in range(10)]
for i in range(1000):
    for x in range(len(inputvals)):
        nn.forward([inputvals[x]/10])
        nn.backward([targetvals[x]/3.16])
    if i%100==0:
        print("iteration:",i)
       
for n in range(len(inputvals)):
     predictedvals[n]=(nn.forward([inputvals[n]/10])[0])*3.16
     print(inputvals[n],predictedvals[n])

plt.figure(figsize=(8, 6))
plt.plot(inputvals, targetvals, marker='o', linestyle='-', color='b', label=r'$y = \sqrt{x}$')
plt.plot(inputvals, predictedvals, marker='s', linestyle='--', color='r', label=r'predicted values')
# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of $y = \sqrt{x}$')

# Show grid and legend
plt.grid(True)
plt.legend()

# Display the graph
plt.show()
#_________________________________________________________still not working

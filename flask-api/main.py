from flask import Flask,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


x=[]
y=[]

n = 0

def pred_y(m,c,x):
    return m*x+c

# return partial diff along m
def grad_dm(m,c):
    s = 0
    for i,j in zip(x,y):
        fx = pred_y(m,c,i)
        s = s + i*(j-fx) 
    s = (-2.0*s)/n  #  sum all the errors and finding partial derivative along m
    return s

# returns partial diff along c
def grad_dc(m,c):
    s = 0
    for i,j in zip(x,y):
        fx = pred_y(m,c,i)
        s = s + (j - fx) 
    s = (-2.0*s)/n # sum all the errors and finding partial derivative along c
    return s

def do_gradient():
    
    m,c,lr,max_iter = 0,0,0.01,10
    dm = 0
    dc = 0
    for i in range(20):
        dm = grad_dm(m,c)
        dc = grad_dc(m,c)
        m = m - lr*dm
        c = c - lr*dc
    return (m,c)


#m1,c1 = do_gradient()


def predict(int_list,pred):
   
   m1,c1 = do_gradient()
   res = pred_y(m1,c1,pred)
   
   return str(res)




@app.route('/')
def hello():
   return 'Hello from Flask!'


@app.route('/predict',methods = ['POST', 'GET'])
def compile():
   global n,x,y
   
   data = request.args.get('data')
   value = int(request.args.get('value'))
   data = data.split(",")
   integer_map = map(int, data)
   integer_list = list(integer_map)

   n = len(integer_list)
   y = integer_list
   x = range(1,n)
   print(x)
   print(y)   
   result = predict(integer_list,value)
   print("----")
   print(result)
   
   return result

if __name__ == '__main__':
   app.run(debug = True,host='0.0.0.0', port=5000)









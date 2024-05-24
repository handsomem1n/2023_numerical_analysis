import numpy as np
def falseposition(func, xl, xu):
maxit=100 # maxit은 최대 반복 횟수
es=1.0e-4 # es는 허용 오차
test=func(xl)*func(xu) # test는 xl과 xu의 함수값의 곱
if test > 0:
print('no sign change') # 근이 존재하지 않는다
return [], [], [], []
iter=0 # 반복 횟수(iter)
xr=xl
ea=100
while(1):
xrold=xr
#xr=np.float((xl+xu)/2)
xr = np.float(xu-func(xu)*(xl-xu)/(func(xl)-func(xu)))
iter=iter+1
if xr != 0:
ea=np.float(np.abs(np.float(xr)-np.float(xrold))/np.float(xr))*100
test=func(xl)*func(xr)
if test > 0:
xl=xr
elif test < 0:
xu=xr
else:
ea=0
if np.int(ea<=es) | np.int(iter>=maxit):
break
root=xr
fx=func(xr)
return root, fx, ea, iter
if __name__ == '__main__':
fm=lambda m: np.sqrt(9.81*m/0.25)*np.tanh(np.sqrt(9.81*0.25/m)*4)-36
root, fx, ea, iter=falseposition(fm, 40, 200)
print('root=', root)
print('f(root)=', fx)
print('ea=', ea)
print('iter=', iter)
if __name__=='__main__':
fm=lambda m: np.sqrt(9.81*m/0.25)*np.tanh(np.sqrt(9.81*0.25/m)*4)-36
root, fx, ea, iter=falseposition(fm, 40, 200)
print('root=', root)
print('f(root)=', fx)
print('ea=', ea)
print('iter=', iter)
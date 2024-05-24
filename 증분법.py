import numpy as np

def incsearch(func, xmin, xmax): #incserach함수 설명 : incsearch 함수는 함수 func와 최소값 xmin과 최대값 xmax을 인자로 받아서, xmin부터 xmax까지 일정한 간격으로 값을 증가시키면서 func의 값이 부호가 바뀌는 구간을 찾아냄.
    x=np.arange(xmin, xmax+1)
    #np.linspace(xmin, xmax, ns)
    f=func(x)
    nb=0 # 이 구간들을 xb 리스트에 저장하고, 총 구간 수를 nb로 반환
    xb=[]

    for k in np.arange(np.size(x)-1):
        if np.sign(f[k]) != np.sign(f[k+1]):  # k=141
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    return nb, xb


g=9.81; cd=0.25; v=36; t=4;
# incsearch 함수를 사용하여 fp라는 함수의 근을 구함
# fp는 입력값 mp에 대해 fp(mp) = sqrt(g*mp/cd)*tanh(sqrt(g*cd/mp)*t) - v의 값을 반환하는 람다 함수
fp = lambda mp:np.sqrt(g*np.array(mp)/cd)*np.tanh(np.sqrt(g*cd/np.array(mp))*t)-v
nb, xb=incsearch(fp, 1, 20) # fp = xmin / 1 = xmax / ns = 200
print('number of brackets= ',nb)
print('root interval=', xb)
# incsearch(fp, 1, 200)를 호출하여 fp 함수에서 근을 찾고, 그 결과를 nb와 xb에 저장함
# 마지막으로 nb와 xb를 출력하겠지

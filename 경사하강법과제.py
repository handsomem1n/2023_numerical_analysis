
import numpy as np
import matplotlib.pyplot as plt #데이터 시각화를 위한 Matplotlib 라이브러리의 pyplot 모듈을 plt라는 이름으로 가져옵

# 데이터 생성 --------------------------------
np.random.seed(seed=1) # 난수를 초기화하고, 난수를 고정
X_min = 4 # X의 하한(표시 용) = 나이의 최솟값
X_max = 30 # X의 상한(표시 용) = 나이의 최댓값
X_n = 16 # X의 상한(표시 용) = 생성할 데이터 갯수 지정
X = 5 + 25 * np.random.rand(X_n) # 나이의 범위가 4에서 30인 난수를 X_n개 생성-> 5에서 30사이의 값이 생성됨.

Prm_c = [170, 108, 0.2] # prm(parameter), 나이에 따른 키를 계산하는 데 사용되는 실제 매개변수.

T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n) # 나이에 따른 키를 계산하고, 가우시안 잡음을 추가. 이렇게 생성된 T는 키 데이터를 나타냄.

np.savez('ch5_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T) # 생성된 데이터를 ch%_data.npz파일에 저장. 이 파일은 나중에 load하여 이 데이터를 재사용 가능.

                # 리스트 5-1-(2)
print(X) # 나이 데이터 X출력

print(np.round(X, 2)) # np.round(X, 2)는 배열 X의 각 요소를 소수점 아래 두 자리까지 반올림 -> 나이 데이터 X를 소숫점 둘째 자리까지 반올림하여 출력

print(np.round(T, 2)) # 키 데이터 T를 소수점 둘째 자리까지 반올림하여 출력

X=np.round(X, 2) # 나이 데이터 X를 소수점 둘째 자리까지 반올림
T=np.round(T, 2) # 키 데이터 T를 소수점 둘째 자리까지 반올림

plt.figure(figsize=(4, 4))
plt.plot(X, T, marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

def mse_line(x, t, w): # 나이, 키, (나이 , w[0]), (키 , w[1]) : 튜플 자료형
    # x=X ; t=T; w= (x0[i0] , x1[i1]) ; w[0] = -25.0; w[1]= 120.0
    y = w[0] * x + w[1] # (나이 , w[0]), (키 , w[1])
    mse = np.mean((y - t)**2) # 실제 키(t)와 예측된 키(y)의 차이의 제곱을 계산
    return mse # mse(Mean Squared Error) : 평균 제곱 오차

xn = 100 # 등고선 표시 해상도
w0_range = [-25, 25] #나이에 대한 기울기 값 범위 설정
w1_range = [120, 170] # 키에 대한 w1 범위 설정
x0 = np.linspace(w0_range[0], w0_range[1], xn) # x0 : 나이를 나타내는 배열 생성
x1 = np.linspace(w1_range[0], w1_range[1], xn) # x1 : 키를 나타내는 배열 생성
xx0, xx1 = np.meshgrid(x0, x1) # xx0, xx1는 x0과 ,x1를 이용하여 전수검사를 위한 격자 형태의 배열 생성
J = np.zeros((len(x0), len(x1))) # 오차함수 J는 mse_line()의 결과를 저장하기 위한 배열로, x0와 x1의 길이를 크기로 갖는 2차원 배열로 초기화됨.
for i0 in range(xn):
    for i1 in range(xn):
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1])) # 튜플 자료형
plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)

ax = plt.subplot(1, 2, 1, projection='3d')

ax.plot_surface(xx0, xx1, J, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='black')


ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.view_init(20, -60)

plt.subplot(1, 2, 2)

cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=[100, 1000, 10000, 100000])

cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)
plt.show()

def dmse_line(x, t, w): # 나이, 키, (나이 , w[0]), (키 , w[1])
    # w_i[i - 1] = [10, 165]
    # w=[10, 165]; x=X; t=T
    y = w[0] * x + w[1]

    d_w0 = 2 * np.mean((y - t) * x) # @@@@@@@@@@@@@@@ d_w0 = w0에서의 기울기임.
    d_w1 = 2 * np.mean(y - t) # @@@@@@@@@@@@@@@@@@@@ d_w1 = w1에서의 기울기임.

    return d_w0, d_w1  # 4666.8  292.6

d_w = dmse_line(X, T, [10, 165]) # dmse_line() 함수를 호출하여 나이(X)와 키(T) 데이터에 대한 가중치 [10, 165]에서의 기울기를 계산

print(np.round(d_w, 1)) # d_w를 소수점 첫째 자리까지 반올림하여 출력 -> 이는 나이(X)와 키(T) 데이터에서의 가중치 [10, 165]에서의 기울기를 나타냄.



## rec
def fit_line_num(x, t):
# 나이(x)와 키(t) 데이터를 입력으로 받음. def fit_line_num(x, t): # 나이(x)와 키(t) 데이터를 입력으로 받음.
    # x=X; t=T
    w_init = [10.0, 165.0] # 초기 매개 변수[10.0, 165.0] 설정 -> 경사하강법의 시작 지점으로 사용됨(이는 나이와 키 데이터에는 존재하지 않는 임의의 값임.)

    alpha = 0.001 # 학습률(learning rate)로서 경사 하강법의 각 스텝 사이의 간격을 결정(이게 곧 스텝사이즈, 간격을 얼마나 두고 반복시행할 건지)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    i_max = 100000 # 최대 반복 횟수를 나타냄
    eps = 0.1 # 반복을 종료할 기울기의 절대 값의 한계를 나타냄
    w_i = np.zeros([i_max, 2]) # 나이 값과 키 값의 변화를 저장하는 그릇
    w_i[0, :] = w_init # w_i 배열의 첫 번째 행은 초기 매개변수 w_init를 저장하고, 이후 반복마다 갱신되는 매개변수를 저장

    for i in range(1, i_max):
        dmse = dmse_line(x, t, w_i[i - 1]) # dmse_line() 함수를 호출하여 나이x와 키t에 대한 기울기를 계산하고, 해당 기울기를 이용하여 매개변수 w_i를 갱신
        # x=X (나이); t=T (키)
        # 𝛻𝑓(𝑥_𝑘 ) 나이에 대해 경사하강법 적용
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0] # 𝛻𝑓(𝑥_𝑘 ) 나이에 대해 경사하강법 적용
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1] # 𝛻𝑓(𝑥_𝑘 ) 키에 대해 경사하강법 적용
        if max(np.absolute(dmse)) < eps: # 종료판정(매 반복마다 기울기의 절대값이 eps보다 작아지면 반복을 종료), np.absolute는 절대치
            print("break i ", i)
            print("break dmse ", dmse)

            break

    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    print("w0*, w1* = ", np.round(w_i[i], 2))

    w_i = w_i[:i, :]

    return w0, w1, dmse, w_i
plt.figure(figsize=(4, 4))
xn = 100

w0_range = [-25, 25] # w0_range와 w1_range는 각각 가중치 w0와 w1의 범위를 설정
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn) #x0와 x1은 w0_range와 w1_range를 이용하여 각각 w0와 w1의 값들을 생성
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1) #xx0와 xx1은 x0와 x1을 이용하여 전수검사를 위한 격자 형태의 배열을 생성
J = np.zeros((len(x0), len(x1))) # J는 오차 함수 mse_line()를 계산하여 평균 제곱 오차를 저장하는 배열
#이중 for문을 통해 모든 x0와 x1의 조합에 대해 평균 제곱 오차를 계산하고 J 배열에 저장
for i0 in range(xn):
    for i1 in range(xn):
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))


# 등고선에 표시
cont = plt.contour(xx0, xx1, J, 30, colors='black', levels=(100, 1000, 10000, 100000))
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True) # 그리드 표시

W0, W1, dMSE, W_history = fit_line_num(X, T) # fit_line_num() 함수를 호출하여 선형 회귀 모델을 학습하고, 최적의 매개변수 W0, W1, 기울기 dMSE, 그리고 매개변수의 변화 W_history를 반환


print('반복 횟수 {0}'.format(W_history.shape[0]))
print('W=[{0:.6f}, {1:.6f}]'.format(W0, W1))
print('dMSE=[{0:.6f}, {1:.6f}]'.format(dMSE[0], dMSE[1]))
print('MSE={0:.6f}'.format(mse_line(X, T, [W0, W1])))

plt.plot(W_history[:, 0], W_history[:, 1], '.-', color='yellow', markersize=10, markeredgecolor='red')

plt.show()
plt.show()

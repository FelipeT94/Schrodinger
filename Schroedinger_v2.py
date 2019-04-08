import math
import numpy as np
import matplotlib.pyplot as plt
nx = 500
m = 10
ni = 100
x1 = -10
x2 = 10
h = (x2-x1)/nx
mh = m*h
ul = np.zeros(nx+1)
ur = np.zeros(nx+1)
ql = np.zeros(nx+1)
qr = np.zeros(nx+1)
s = np.zeros(nx+1)
u = np.zeros(nx+1)
Del = 1*10**-6
de = .1
delta = .2
def simpson(y,h):
    n = len(y)-1
    s0 = 0
    s1 = 0
    s2 = 0
    for i in range(1,n,2):
        s0 +=y[i]
        s1 +=y[i-1]
        s2 +=y[i+1]
    s = (s1+4*s0+s2)/3
    if (n+1)%2 ==0:
        return h*(s+(5*y[n]+8*y[n-1]-y[n-2])/12)
    else:
        return h*s
def numerov(m,h,q,s, u0 = 0, u1 = .1):
    u =np.zeros(m)
    u[0] = u0
    u[1] = u1
    g = h*h/12
    for i in range(1,m-1):
        c0 = 1+g*q[i-1]
        c1 = 2-10*g*q[i]
        c2 = 1+g*q[i+1]
        d = g*(s[i+1]+s[i-1]+10*s[i]) # 0
        u[i+1] = (c1*u[i]-c0*u[i-1]+d)/c2
    return u
def potential(alpha,Lambda, w, m, f_type):
    k = w**2*m
    if f_type == 'c':
        v = alpha*alpha*Lambda*(Lambda-1)*(.5-1/np.power(np.cosh(alpha*x),2))/2
    elif f_type =='o':
        v = .5*k*np.power(x,2)
    return v
def x_array(xmin,xmax,xn):
    x = np.linspace(xmin,xmax,xn)
    return x
def secant(n,Del,x,dx,V_i):
    k = 0
    x1 = x+dx
    while abs(dx)>Del and k<n:
        d = f(x1,V_i)-f(x,V_i)
        x2 = x1-f(x1,V_i)*(x1-x)/d
        x = x1
        x1 = x2
        if (abs(x1-x)>delta):
   #         print('x1_p = ',x1)
  #          print('x = ',x)
            x1=np.sign(x1-x)*delta+x
 #           print('x1 = ',x1)
        dx = x1-x
#        print('dx = ',dx)
        k +=1
    if k ==n:
        print("Convergence not found after ", n, " iterations")
    return x1
def f(x,V_i):
    (nr,nl,ur,ul) = wave(x,V_i)
    f0 = ur[nr-1]+ul[nl-1]-ur[nr-3]-ul[nl-3]
    return f0/(2*h*ur[nr-2])
def wave(energy,V_i):
    y = np.zeros(nx+1)
    #set up function q(x) in the equation
    ql = 2*(energy-V_i)
    for i in range(0,nx+1):
        qr[nx-i] = ql[i]
    #find the matching point at the right turning point
    '''
    problems finding turning point at given different potentials
    debug for im, ql, qr
    ql[i]*ql[i+1] must have opposite sings and ql[i]>0
    '''
    im = int((nx+1)/2)
    for i in range(0,nx+1):
        if (((ql[i]*ql[i+1])<0) and (ql[i]>0)):
            im = i
            break
        if ((ql[i]*ql[i+1]) == 0):
            im = i
            print('test = ', im)
            break
    #carry out the numerov integrations
    nl = im+2
    #print('nl = ',nl)
    nr = nx-im+2
    #print('nr = ',nr)
    ul = numerov(nl,h,ql,s)
    ur = numerov(nr,h,qr,s)
    ratio = ur[nr-2]/ul[im]
    for i in range(0,im+1):
        u[i] = ratio*ul[i]
        y[i] = u[i]*u[i]
    ul[nl-1] *=ratio
    ul[nl-3] *=ratio
    for i in range(0,nr-1):
        u[i+im] = ur[nr-i-2]
        y[i+im] = u[i+im]*u[i+im]
    sum = simpson(y,h)
    sum = math.sqrt(sum)
    for i in range(0,nx+1):
        u[i] /= sum
    return nr,nl,ur,ul
if __name__ == "__main__":
    x = x_array(-10,10,501)
    v_i = 10
    V, eigenvalues,e_state,e_initial = [], [], [], []
    a = 1
    l = 4
    m = 1
    w = 1
    num_i = 5
    V.append(x)
    for i in range (0,v_i):
        f_type = 'o'
        V_temp = potential(a,l,w,m,f_type)
        V.append(V_temp)
        w += 1       #variable used to produce multiple potentials
    temp = 1
    V_i = V[temp]
    v_temp = np.array(V)
    temp_1 = v[0]
    V[][]
    np.savetxt('')
    plt.subplot(2,1,2)
    plt.plot(x,V[temp])
    upper_bound = np.max(V[temp]) -.4
    lower_bound = np.min(V[temp])
    #for i in range(1,v_i)    #eigenstates while varying potential
    for e in np.linspace(lower_bound,upper_bound/10,num_i):
        e_initial.append(e)
        e = secant(ni,Del,e,de,V_i)
        plt.subplot(2,1,1)
        plt.plot(x,u)
        count = 0
        for i in range(0,nx):
            if u[i]*u[i+1] < 0:
                count +=1
        e_state.append(count)
        eigenvalues.append(e)
    plt.savefig('test.png')
# =============================================================================
#     print("the eigenvalue: ", eigenvalues)
#     print("the eigen states: ", e_state)
#     print("the initial eigen value was: ", e_initial)
# =============================================================================
    print('E_state   E_initial   Eigenvalue')
    for i in range(num_i):
        print("{:4d} {:12.4f} {:12.4f}".format(e_state[i], e_initial[i], eigenvalues[i]))
    
    

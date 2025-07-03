import numpy as np
import matplotlib.pyplot as plt
#Second order, 2 stage RK
#Let's solve the problem from HW5
#pick some values
cd = 0.25
m = 68.1
g = 9.81
t0 = 0.0
tf = 10.0
v0 = 0.0
x0 = 0.0
#stability for forward euler is dt<2/a
dt = 1.0
nt = int(tf/dt)
t = np.linspace(t0,tf,nt+1)

def rhs_func_x(v):
    return v

def rhs_func_v(v):
    return g-cd/m*v**2

def RK2(t,x,v,dt):
    k1_x = rhs_func_x(v)

    k1_v = rhs_func_v(v)
    k2_x = rhs_func_x(v+k1_v*dt)
    k2_v = rhs_func_v(v+k1_v*dt)
    x_next = x + dt*(0.5*k1_x + 0.5*k2_x)
    v_next = v + dt*(0.5*k1_v + 0.5*k2_v)
    return x_next,v_next

x_RK2 = np.zeros(nt+1)
v_RK2 = np.zeros(nt+1)
x_RK2[0] = x0
v_RK2[0] = v0

for i in range(nt):
    x_RK2[i+1], v_RK2[i+1] = RK2(t[i],x_RK2[i],v_RK2[i],dt)

#compute analytic solution for comparison
x_true = m/cd*np.log(np.cosh(np.sqrt(g*cd/m)*t))
v_true = np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)

#plot what we get
#compare by plotting
plt.plot(t,v_true,label='true velocity')
plt.plot(t,v_RK2,'--',label='RK2 velocity')
plt.grid()
plt.xlabel('t(s)')
plt.ylabel('velocity (m/s)')
plt.title('RK2 approximation for velocity')
#plt.ylim([0.0,1.0])
plt.legend()
plt.show()
plt.close()
#plot what we get
#compare by plotting
plt.plot(t,x_true,label='true position')
plt.plot(t,x_RK2,'--',label='RK2 position')
plt.grid()
plt.xlabel('t(s)')
plt.ylabel('position (m)')
plt.title('RK2 approximation for position')
#plt.ylim([0.0,1.0])
plt.legend()
plt.show()
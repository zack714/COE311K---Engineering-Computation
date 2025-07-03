import numpy as np
import matplotlib.pyplot as plt

#variables to set up problem

#initial condition
v0 = 0
t0 = 0
tf = 10
g = 9.81
cd = 0.25
m = 68.1



dt = 5.0


#number of time steps I need to preform 
nt = int((tf-t0)/dt)

#create a vector to store the solution, including the initial state
v = np.zeros(nt+1)
v[0]= v0


#time loop
for i in range(nt):
    v[i+1] = v[i] + dt*(g-(cd/m)*(v[i]**2))

t = np.linspace(t0,tf,nt+1)
v_analytic = np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)
"""
#plot the difference between the true and forward euler solution
plt.plot(t,v,label = "Forward Euler Solution")
plt.plot(t,v_analytic,'---',label = "Exact Solution")
plt.xlabel("Time (sec)")
plt.ylabel("Velocity (m/s)")
plt.title("Velcocity of a free falling object subject to drag")
plt.grid()
plt.legend()
plt.show()
"""
#stability demonstration w/Linear ODE
v0 = 1.0
t0 = 0
tf = 5.0
dt = 0.1
a = 10.0

#number of time steps I need to preform 
nt = int((tf-t0)/dt)
#create a vector to store the solution, including the initial state
v_implicit_Euler = np.zeros(nt+1)
v_explicit_Euler = np.zeros(nt+1)
v_CN = np.zeros(nt+1)
v_implicit_Euler[0] = v0
v_explicit_Euler = v0
v_CN[0] = v0
#create a vector for time 
t = np.linspace(t0,tf,nt+1)

#time loop
for i in range(nt):
    v_implicit_Euler[i+1] = (1.0/(1.0+dt*a))*v_implicit_Euler[i]
    v_explicit_Euler[i+1] = v_explicit_Euler[i]+dt*a*v_explicit_Euler[i]
    v_CN[i+1] = ((1.0-dt*a/2.0)/(1.0+dt*a/2.0)) * v_CN[i]

    v_analytic = v0*np.exp(-a*t)

#plot the difference between the true and forward euler solution
plt.plot(t,v_explicit_Euler,label = "Explicit Euler")
plt.plot(t,v_analytic,'---',label = "Exact Solution")
plt.xlabel("Time (sec)")
plt.ylabel("Velocity (m/s)")
plt.title("Velcocity of a free falling object subject to drag")
plt.grid()
plt.legend()
plt.show()


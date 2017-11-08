
"""
This code compiles the following utility functions and methods:
- Runge-Kutta (4th order) ODE solver
"""

# Numerical method
def rk4(f, t, X, V):
    ''' Solves the ODE using 4th order Runge-Kutta '''
    k1 = f(t, X, V)
    k2 = f(t+(dt/2), X+(dt*k1/2), V)
    k3 = f(t+(dt/2), X+(dt*k2/2), V)
    k4 = f(t+dt, X+(dt*k3), V)
    y = X + dt*(k1 + (2*k2) + (2*k3) + k4)/6

    return y
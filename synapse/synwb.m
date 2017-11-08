function y=synwb(t,r,V)
    global tau_d;
    global Vo;
    y=12*(1-r)./(1+exp(-(V-Vo)/2))-r./tau_d;
return
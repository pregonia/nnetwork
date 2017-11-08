%Simulation Parameters
    global dt;             dt=0.1;
                           Tf=1;
                           N=ceil(Tf*1000/dt);
    global tau_r;          tau_r=0.5;
    global Vo;             Vo=0;
                           gsyn=0.1;
                           Esyn=-75;
    
%Membrane Potentials
    V=-40+30*(sin(10*[0:2*pi/(N-1):2*pi])+sin(7*[0:2*pi/(N-1):2*pi]).^2);
    S=zeros(size(V));
    Ix=S;
    
for i=1:N-1
    t=i*dt;
    S(i+1)=rk4(@synwb,t,S(i),V(i));
    Ix(i+1)=gsyn*S(i+1)*(V(i)-Esyn);
end


%Plot
    subplot(311); plot(dt*[0:N-1],V); 
    subplot(312); plot(dt*[0:N-1],S);
    subplot(313); plot(dt*[0:N-1],-Ix);
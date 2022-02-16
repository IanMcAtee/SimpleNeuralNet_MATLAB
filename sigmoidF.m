%FUNCTION: sigmoidF.m
%AUTHOR: Ian McAtee
%DATE: 11/21/2021
%DESCRIPTION: Function to compute the sigmoid funcion
%INPUT:
    %net: A dx1 vector that can represent the weighted input to a neural
          %network node
%OUTPUT: 
    %sigNet: A dx1 vector containing the sigmoid of the net input

function sigNet = sigmoidF(net)
    a = 1.7159;
    beta = 2/3;
    netLen = length(net);
    sigNet = zeros(netLen,1);
    for i = 1:netLen
        sigNet(i) = a*tanh(beta*net(i));
    end
end
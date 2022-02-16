%FUNCTION: sigD.m
%AUTHOR: Ian McAtee
%DATE: 11/21/2021
%DESCRIPTION: Function to compute the derivative of the sigmoid funcion
%INPUT:
    %net: A dx1 vector that can represent the weighted input to a neural
          %network node
%OUTPUT: 
    %sigD: A dx1 vector containing the sigmoid derivative of the net input

function sigD = sigDeriv(net)
    a = 1.7159;
    beta = 2/3;
    netLen = length(net);
    sigD = zeros(netLen,1);
    for d = 1:netLen
        sigD(d) = (beta/a)*(a+sigmoidF(net(d)))*(a-sigmoidF(net(d)));
    end
end
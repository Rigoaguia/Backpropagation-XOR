clear;
clc;

%XOR
% x1  x2  d
% 0   0   0
% 0   1   1
% 1   0   1
% 1   1   0

x1 = [0 0 1 1];  %x1
x2 = [0 1 0 1];  %x2     

d  = [ 0  1  1  0 ]; %saída esperada
% pesos
w13 = rand(1,1);
w23 = rand(1,1);
w14 = rand(1,1);
w24 = rand(1,1);
w35 = rand(1,1);
w45 = rand(1,1);
%pesos bias
b3 = rand(1,1);
b4 = rand(1,1);
b5 = rand(1,1);

p = 4; % qt de amostras
n = 0.0025; % tx de aprendizado
pr = 0.01; % precisão
e = 1;
epoca = 0;

% fase de treinamento 

while (abs(e) > pr)
    
    for i=1:p
        
        y3 = sigmoid(x1(i)*w13 + x2(i)*w23 + b3);
        y4 = sigmoid(x1(i)*w14 + x2(i)*w24 + b4);
        y5(i) = sigmoid(y3*w35 + y4*w45 + b5);
        e  = (d(i) - y5(i));
        % erro gradiente no neuronio 5
        s5 = y5(i)*(1-y5(i))*e;
        dw35 = n*y3*s5;
        dw45 = n*y4*s5;
        db5 = n*(+1)*s5;
        % erro gradiente no neuronio 3
        s3 = y3*(1-y3)*s5*w35;
        dw13 = n*x1(i)*s3;
        dw23 = n*x2(i)*s3;
        db3 = n*(+1)*s3;
        % erro gradiente no neuronio 4
        s4 = y4*(1-y4)*s5*w45;
        dw14 = n*x1(i)*s4;
        dw24 = n*x2(i)*s4;
        db4 = n*(1)*s4;
        % ajuste
        w13 = w13 + dw13;
        w14 = w14 + dw14;
        w23 = w23 + dw23;
        w24 = w24 + dw24;
        w35 = w35 + dw35;
        w45 = w45 + dw45;
        b3 = b3 + db3;
        b4 = b4 + db4;
        b5 = b5 + db5;   
    end
    epoca = epoca + 1;
    
end
epoca
y5





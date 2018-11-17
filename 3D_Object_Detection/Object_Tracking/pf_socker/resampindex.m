

function indices = resampindex(weights)

weights = max(0,weights);
weights = weights/sum(weights);
N = length(weights);
cumprob=[0 cumsum(weights)];
indices = zeros(1,N);

if (0)
%usual version where each sample drawn randomly
uni=rand(1,N);
for j=1:N
  ind=find((uni>cumprob(j)) & (uni<=cumprob(j+1)));
  indices(ind)=j;
end
return
end

%more efficient version where one random sample seeds
%a deterministically methodical sampling by 1/N
i=1;
u1 = rand(1)/N;
for j=1:N
    uj = u1 + (j-1)/N;
    while (uj > cumprob(i))
        i=i+1;
    end
    indices(j) = (i-1);
end
return


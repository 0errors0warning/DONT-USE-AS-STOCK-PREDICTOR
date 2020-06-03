B = load('LSTM2.csv')
B=B(:,2)
xx=zeros(3000,12)
for i=1:3000
    for j=1:12
        xx(i,j)=B(i+j);
    end;
end;
[N,~]=size(xx);
ntrain=round(N-200); 
mm=20; 
for i=1:ntrain
    for j=1:3 
        x11(i,j)=xx(i,j);
        x12(i,j)=xx(i,j+1);
        x13(i,j)=xx(i,j+2);
        x14(i,j)=xx(i,j+3);
        x15(i,j)=xx(i,j+4);
        x16(i,j)=xx(i,j+5);
        x17(i,j)=xx(i,j+6);
        x18(i,j)=xx(i,j+7);
        x19(i,j)=xx(i,j+8);
    end; 
    y(i,1)=xx(i,12); 
end;
[zb11 rg11]=wmdeepzb(mm,x11,y);
[zb12 rg12]=wmdeepzb(mm,x12,y);
[zb13 rg13]=wmdeepzb(mm,x13,y);
[zb14 rg14]=wmdeepzb(mm,x14,y);
[zb15 rg15]=wmdeepzb(mm,x15,y);
[zb16 rg16]=wmdeepzb(mm,x16,y);
[zb17 rg17]=wmdeepzb(mm,x17,y);
[zb18 rg18]=wmdeepzb(mm,x18,y);
[zb19 rg19]=wmdeepzb(mm,x19,y);
x21(1:ntrain,1)=wmdeepyy(mm,zb11,rg11,x11);
x21(1:ntrain,2)=wmdeepyy(mm,zb12,rg12,x12);
x21(1:ntrain,3)=wmdeepyy(mm,zb13,rg13,x13);
x22(1:ntrain,1)=x21(:,2);
x22(1:ntrain,2)=x21(:,3);
x22(1:ntrain,3)=wmdeepyy(mm,zb14,rg14,x14);
x23(1:ntrain,1)=x21(:,3);
x23(1:ntrain,2)=x22(:,3);
x23(1:ntrain,3)=wmdeepyy(mm,zb15,rg15,x15);
x24(1:ntrain,1)=x22(:,3);
x24(1:ntrain,2)=x23(:,3);
x24(1:ntrain,3)=wmdeepyy(mm,zb16,rg16,x16);
x25(1:ntrain,1)=x23(:,3);
x25(1:ntrain,2)=x24(:,3);
x25(1:ntrain,3)=wmdeepyy(mm,zb17,rg17,x17);
x26(1:ntrain,1)=x24(:,3);
x26(1:ntrain,2)=x25(:,3);
x26(1:ntrain,3)=wmdeepyy(mm,zb18,rg18,x18);
x27(1:ntrain,1)=x25(:,3);
x27(1:ntrain,2)=x26(:,3);
x27(1:ntrain,3)=wmdeepyy(mm,zb19,rg19,x19);
[zb21 rg21]=wmdeepzb(mm,x21,y);
[zb22 rg22]=wmdeepzb(mm,x22,y);
[zb23 rg23]=wmdeepzb(mm,x23,y);
[zb24 rg24]=wmdeepzb(mm,x24,y);
[zb25 rg25]=wmdeepzb(mm,x25,y);
[zb26 rg26]=wmdeepzb(mm,x26,y);
[zb27 rg27]=wmdeepzb(mm,x27,y);
x31(1:ntrain,1)=wmdeepyy(mm,zb21,rg21,x21);
x31(1:ntrain,2)=wmdeepyy(mm,zb22,rg22,x22);
x31(1:ntrain,3)=wmdeepyy(mm,zb23,rg23,x23);
x32(1:ntrain,1)=x31(:,2);
x32(1:ntrain,2)=x31(:,3);
x32(1:ntrain,3)=wmdeepyy(mm,zb24,rg24,x24);
x33(1:ntrain,1)=x31(:,3);
x33(1:ntrain,2)=x32(:,3);
x33(1:ntrain,3)=wmdeepyy(mm,zb25,rg25,x25);
x34(1:ntrain,1)=x32(:,3);
x34(1:ntrain,2)=x33(:,3);
x34(1:ntrain,3)=wmdeepyy(mm,zb26,rg26,x26);
x35(1:ntrain,1)=x33(:,3);
x35(1:ntrain,2)=x34(:,3);
x35(1:ntrain,3)=wmdeepyy(mm,zb27,rg27,x27);
[zb31 rg31]=wmdeepzb(mm,x31,y);
[zb32 rg32]=wmdeepzb(mm,x32,y);
[zb33 rg33]=wmdeepzb(mm,x33,y);
[zb34 rg34]=wmdeepzb(mm,x34,y);
[zb35 rg35]=wmdeepzb(mm,x35,y);
x41(1:ntrain,1)=wmdeepyy(mm,zb31,rg31,x31);
x41(1:ntrain,2)=wmdeepyy(mm,zb32,rg32,x32);
x41(1:ntrain,3)=wmdeepyy(mm,zb33,rg33,x33);
x42(1:ntrain,1)=x41(:,2);
x42(1:ntrain,2)=x41(:,3);
x42(1:ntrain,3)=wmdeepyy(mm,zb34,rg34,x34);
x43(1:ntrain,1)=x41(:,3);
x43(1:ntrain,2)=x42(:,3);
x43(1:ntrain,3)=wmdeepyy(mm,zb35,rg35,x35);
[zb41 rg41]=wmdeepzb(mm,x41,y);
[zb42 rg42]=wmdeepzb(mm,x42,y);
[zb43 rg43]=wmdeepzb(mm,x43,y);
x51(1:ntrain,1)=wmdeepyy(mm,zb41,rg41,x41);
x51(1:ntrain,2)=wmdeepyy(mm,zb42,rg42,x42);
x51(1:ntrain,3)=wmdeepyy(mm,zb43,rg43,x43);
[zb51 rg51]=wmdeepzb(mm,x51,y);
for i=1:N
    for j=1:3
        x11(i,j)=xx(i,j);
        x12(i,j)=xx(i,j+1);
        x13(i,j)=xx(i,j+2);
        x14(i,j)=xx(i,j+3);
        x15(i,j)=xx(i,j+4);
        x16(i,j)=xx(i,j+5);
        x17(i,j)=xx(i,j+6);
        x18(i,j)=xx(i,j+7);
        x19(i,j)=xx(i,j+8);
    end;
    y(i,1)=xx(i,12);
end;
x21(1:N,1)=wmdeepyy(mm,zb11,rg11,x11);
x21(1:N,2)=wmdeepyy(mm,zb12,rg12,x12);
x21(1:N,3)=wmdeepyy(mm,zb13,rg13,x13);
x22(1:N,1)=x21(:,2);
x22(1:N,2)=x21(:,3);
x22(1:N,3)=wmdeepyy(mm,zb14,rg14,x14);
x23(1:N,1)=x21(:,3);
x23(1:N,2)=x22(:,3);
x23(1:N,3)=wmdeepyy(mm,zb15,rg15,x15);
x24(1:N,1)=x22(:,3);
x24(1:N,2)=x23(:,3);
x24(1:N,3)=wmdeepyy(mm,zb16,rg16,x16);
x25(1:N,1)=x23(:,3);
x25(1:N,2)=x24(:,3);
x25(1:N,3)=wmdeepyy(mm,zb17,rg17,x17);
x26(1:N,1)=x24(:,3);
x26(1:N,2)=x25(:,3);
x26(1:N,3)=wmdeepyy(mm,zb18,rg18,x18);
x27(1:N,1)=x25(:,3);
x27(1:N,2)=x26(:,3);
x27(1:N,3)=wmdeepyy(mm,zb19,rg19,x19);
x31(1:N,1)=wmdeepyy(mm,zb21,rg21,x21);
x31(1:N,2)=wmdeepyy(mm,zb22,rg22,x22);
x31(1:N,3)=wmdeepyy(mm,zb23,rg23,x23);
x32(1:N,1)=x31(:,2);
x32(1:N,2)=x31(:,3);
x32(1:N,3)=wmdeepyy(mm,zb24,rg24,x24);
x33(1:N,1)=x31(:,3);
x33(1:N,2)=x32(:,3);
x33(1:N,3)=wmdeepyy(mm,zb25,rg25,x25);
x34(1:N,1)=x32(:,3);
x34(1:N,2)=x33(:,3);
x34(1:N,3)=wmdeepyy(mm,zb26,rg26,x26);
x35(1:N,1)=x33(:,3);
x35(1:N,2)=x34(:,3);
x35(1:N,3)=wmdeepyy(mm,zb27,rg27,x27);
x41(1:N,1)=wmdeepyy(mm,zb31,rg31,x31);
x41(1:N,2)=wmdeepyy(mm,zb32,rg32,x32);
x41(1:N,3)=wmdeepyy(mm,zb33,rg33,x33);
x42(1:N,1)=x41(:,2);
x42(1:N,2)=x41(:,3);
x42(1:N,3)=wmdeepyy(mm,zb34,rg34,x34);
x43(1:N,1)=x41(:,3);
x43(1:N,2)=x42(:,3);
x43(1:N,3)=wmdeepyy(mm,zb35,rg35,x35);
x51(1:N,1)=wmdeepyy(mm,zb41,rg41,x41);
x51(1:N,2)=wmdeepyy(mm,zb42,rg42,x42);
x51(1:N,3)=wmdeepyy(mm,zb43,rg43,x43);
yy=wmdeepyy(mm,zb51,rg51,x51); 
yy=yy'
yy=int16(yy)
plot(yy(ntrain:N),'b')
hold on 
plot(B(ntrain+13:N+60),'r')
clear all

function [zb, ranges]=wmdeepzb(mm,xx,y)
% Train fuzzy system with input xx and output y to get zb and ranges,
% where zb is the in FS (7) and ranges is the endpoints in (11).
extra=[xx,y];
[numSamples,m]=size(extra);
numInput=m-1;
for i=1:numInput
    fnCounts(i)=mm;
end;
ranges = zeros(numInput,2);
activFns = zeros(numInput,2);
activGrades = zeros(numInput,2);
searchPath = zeros(numInput,2);
numCells = 1; % number of regions (cells)
for i = 1:numInput
    a=min(extra(:,2));
    b=max(extra(:,2));
    ranges(i,1) = min(extra(:,i));
    ranges(i,2) = max(extra(:,i));
    numCells = numCells * fnCounts(i);
end;
baseCount(1)=1;
for i=2:numInput
    baseCount(i)=1;
    for j=2:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+2);
    end;
end;
% Generate rules for cells covered by data
zb = zeros(1,numCells); % THEN part centers of generated rules
ym = zeros(1,numCells);
for k = 1:numSamples
    for i = 1:numInput
        numFns = fnCounts(i);%the number of fuzzy sets
        nthActive = 1;%which one to activte
        for nthFn = 1:numFns
            grade = meb2(numFns,nthFn,extra(k,i),ranges(i,1),ranges(i,2));
            if grade > 0
                activFns(i,nthActive) = nthFn;% which fuzzy set own weights [1,20]
                activGrades(i,nthActive) = grade;
                nthActive = nthActive + 1;
            end;
        end; % endfor nthFn
    end; % endfor i
    for i=1:numInput
        if activGrades(i,1) >= activGrades(i,2)%select the biggest weight
            searchPath(i,1)=activFns(i,1);%the ith input variable
            searchPath(i,2)=activGrades(i,1);
        else
            searchPath(i,1)=activFns(i,2);
            searchPath(i,2)=activGrades(i,2);
        end;
    end;
    indexcell=1;
    grade=1;
    for i=1:numInput
        grade=grade*searchPath(i,2);%searchPath(i,2) save the activGrades
        indexcell=indexcell+(searchPath(numInput-i+1,1)-1)*baseCount(i);%计算集合在空间中的具体位置
    end;
    ym(indexcell)=ym(indexcell)+grade;%线，面，片
    zb(indexcell)=zb(indexcell)+extra(k,numInput+1)*grade;%用y来更新权重
end; % endfor k
for j=1:numCells
    if ym(j) ~= 0
        zb(j)=zb(j)/ym(j);
    end;
end;
% Extrapolate the rules to all the cells
for i=1:numInput-1
    baseCount(i)=1;
    for j=1:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+1);
    end;
end;
ct=1;
zbb = zeros(1,numCells);
ymm = zeros(1,numCells);
while ct > 0
    ct=0;
    for s=1:numCells
        if ym(s) == 0
            ct=ct+1;
        end;
    end;
    for s=1:numCells
        if ym(s) == 0
            s1=s;
            index = ones(1,numInput);
            for i=numInput-1:-1:1
                while s1 > baseCount(i)
                    s1=s1-baseCount(i);
                    index(numInput-i)=index(numInput-i)+1;
                end;
            end;
            index(numInput)=s1;
            zbnum=0;
            for i=1:numInput-1
                if index(i) > 1
                    zbb(s)=zbb(s)+zb(s-baseCount(numInput-i));
                    ymm(s)=ymm(s)+ym(s-baseCount(numInput-i));
                    zbnum=zbnum+sign(ym(s-baseCount(numInput-i)));
                end;
                if index(i) < fnCounts(i)
                    zbb(s)=zbb(s)+zb(s+baseCount(numInput-i));
                    ymm(s)=ymm(s)+ym(s+baseCount(numInput-i));
                    zbnum=zbnum+sign(ym(s+baseCount(numInput-i)));
                end;
            end;
            if index(numInput) > 1
                zbb(s)=zbb(s)+zb(s-1);
                ymm(s)=ymm(s)+ym(s-1);
                zbnum=zbnum+sign(ym(s-1));
            end;
            if index(numInput) < fnCounts(numInput)
                zbb(s)=zbb(s)+zb(s+1);
                ymm(s)=ymm(s)+ym(s+1);
                zbnum=zbnum+sign(ym(s+1));
            end;
            if zbnum >= 1
                zbb(s)=zbb(s)/zbnum;
                ymm(s)=ymm(s)/zbnum;
            end;
        end; % endif ym
    end; % endfor s
    for s=1:numCells
        if ym(s) == 0 & ymm(s) ~= 0
            zb(s)=zbb(s);
            ym(s)=ymm(s);
        end;
    end;
end; % endwhile ct
end
function yy=wmdeepyy(mm,zb,ranges,xx)
% Compute the output of fuzzy system with zb and ranges for input xx.
exapp=xx;
[numSamples,m]=size(exapp);
numInput=m;
for i=1:numInput
    fnCounts(i)=mm;
end;
activFns = zeros(numInput,2);
activGrades = zeros(numInput,2);
baseCount(1)=1;
for i=2:numInput
    baseCount(i)=1;
    for j=2:i
        baseCount(i)=baseCount(i)*fnCounts(numInput-j+2);
    end;
end;
for j=1:numInput
    for i1=1:2^j
        for i2=1:2^(numInput-j)
            ma(i2+(i1-1)*2^(numInput-j),j)=mod(i1-1,2)+1;
        end;
    end;
end;
e1sum=0;
for k = 1:numSamples
    for i = 1:numInput
        numFns = fnCounts(i);
        nthActive = 1;
        for nthFn = 1:numFns
            grade = meb2(numFns,nthFn,exapp(k,i),ranges(i,1),ranges(i,2));
            if grade > 0
                activFns(i,nthActive) = nthFn;
                activGrades(i,nthActive) = grade;
                nthActive = nthActive + 1;
            end;
        end;
    end;
    for i=1:numInput
        nn(i,1)=activFns(i,1);
        nn(i,2)=nn(i,1)+1;
        if nn(i,1) == fnCounts(i)
            nn(i,2)=nn(i,1);%找末端
        end;
    end;
    a=0;
    b=0;
    for i=1:2^numInput
        indexcell=1;
        grade=1;
        for j=1:numInput
            grade=grade*activGrades(j,ma(i,j));
            indexcell=indexcell+(nn(numInput-j+1,ma(i,numInput-j+1))-1)*baseCount(j);
        end;
        a=a+zb(indexcell)*grade;
        b=b+grade;
    end;
    yy(k)= a/b; % the fuzzy system output
end; % endfor k
end
function y=meb2(n,i,x,xmin,xmax)
% Compute the value of the i’th membership function in Fig. 2 at x.
h=(xmax-xmin)/(n-1);
if i==1
    if x < xmin
        y=1;
    end;
    if x >= xmin & x < xmin+h
        y=(xmin-x+h)/h;
    end;
    if x >= xmin+h
        y=0;
    end;
end;
if i > 1 & i < n
    if x < xmin+(i-2)*h | x > xmin+i*h
        y=0;
    end;
    if x >= xmin+(i-2)*h & x < xmin+(i-1)*h
        y=(x-xmin-(i-2)*h)/h;
    end;
    if x >= xmin+(i-1)*h & x <= xmin+i*h
        y=(-x+xmin+i*h)/h;
    end;
end;
if i==n
    if x < xmax-h
        y=0;
    end;
    if x >= xmax-h & x < xmax
        y=(-xmax+x+h)/h;
    end;
    if x >= xmax
        y=1;
    end;
end;
end
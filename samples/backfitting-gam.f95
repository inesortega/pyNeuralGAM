nvar=1
nh=30
kernel=1

maxit=10
F=0

call Med_y_Var(Y,W,n,Med,aux)

Eta=med     ! incialmente calculamos los residuos como la medida de y 

if (nvar.eq.1) maxit=1

do nit=1,maxit

    Eta0=Eta    

    do j=1,nvar !for each feature
        do i=1,n
            Eta(i)=Eta(i)-F(i,1,j)
            Z(i)=Y(i)-Eta(i)
        end do
        call Med_y_Var(Z,W,n,Med2,aux)
        
        do i=1,n
            Z(i)=Z(i)-med2
        end do
        
        h=-1
        call rfast_h(X(1,j),Z,W,n,h(j),p,Xb,Pb,kbin,kernel,nh)
        print *, h(j)

        do m=1,2
            call interpola(Xb,Pb(1,m),kbin,X(1,j),F(1,m,j),n)
        end do

        !open (1,file='mierdab.dat')
        !write (1,'(100(a10,1x))') 'x','y'
        !do i=1,kbin
        ! write (1,'(100(f10.4,1x))') xb(i),yb(i),pb(i) !,x(i,2)
        !end do
        !close(1)
        
        
        !stop
        
        !solo recentro las estimaciones, no derivadas!!!

        call Med_y_Var(F(1,1,j),W,n,mu,aux)
        do i=1,n
            F(i,1,j)=F(i,1,j)-mu
        end do

        do i=1,n
            Eta(i)=Eta(i)+F(i,1,j)
        end do
    
    end do ! cierra j

    !calculamos diferencias en la prediccion para cada iter
    err=0
    err2=0
    sumeta0=0
    sumeta=0
    do i=1,n
        sumeta0=sumeta0+eta0(i)
        sumeta=sumeta+eta(i)
        err2=err2+( abs(eta0(i)-eta(i))/abs(eta0(i)) )
    end do
    
    err=(abs(sumeta0-sumeta)/abs(sumeta0) )*100
    
    print *,nit
    err2=(err2/n)*100
    if(err2.le.1) then
        goto 22
    end if

end do ! cierra nit
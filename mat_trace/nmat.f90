program fmat
    use cudafor
    use mpi
    implicit none
    integer :: i, j, k, l, m
    integer,parameter :: nvpm  = 100
    integer,parameter :: ncfgs = 100
    real(8)           :: phimat(nvpm,ncfgs,ncfgs)
    real(8)           :: fmat1(nvpm,nvpm),fmat1_tmp(nvpm,nvpm)
    real(8)           :: rr
    real(8) :: start_time, end_time
    !
    integer(8),allocatable, device   :: fmat_dev(:,:)
    integer(8), device   :: fmat_sum_dev 
    integer(8),allocatable, device   :: phimat_dev(:,:,:)
    integer(8), device   :: r_tmp
    !
    type(cudaEvent)   :: startEvent, stopEvent
    real    :: time
    integer :: istat
    !
    ! mpi
    integer :: numprocs 
    integer :: myid,cnt_mpi
    integer :: ierr
    integer :: namelen
    character*(MPI_MAX_PROCESSOR_NAME) processor_name

    !mpi
    call MPI_Init(ierr)
    call MPI_COMM_RANK(mpi_comm_world, myid, ierr)
    call MPI_COMM_SIZE(mpi_comm_world, numprocs, ierr)
    call MPI_GET_PROCESSOR_NAME(processor_name,namelen,ierr)
    call MPI_BARRIER(mpi_comm_world,ierr)
    write(*,*)'mpi_init done.'
    ! creat time event
    istat = cudaeventcreate(startEvent) 
    istat = cudaeventcreate(stopEvent) 
    !
    ! assign the phimat and fmat
    phimat = 0.0
    fmat1  = 0.0
    do i = 1, nvpm
        do j = 1, ncfgs
            phimat(i,j,j) = 1.0
        enddo
    enddo

    call cpu_time(start_time)
    write(*,*)'numprocs =', numprocs
    do i = 1,nvpm
        do j = 1,nvpm
            cnt_mpi = (i-1)*nvpm + j
            if(mod(cnt_mpi,numprocs) .eq. myid)then
               rr = 0.0
               do k = 1, ncfgs
                   do l = 1, ncfgs
                       rr = rr + phimat(i,k,l)*phimat(j,k,l)
                   enddo 
               enddo 
               fmat1(i,j) = rr
           endif 
        enddo 
    enddo 
    fmat1_tmp = 0
    if(myid .eq. 0)write(*,*)'before allreduce'
    call MPI_ALLREDUCE(fmat1,fmat1_tmp,nvpm*nvpm,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,ierr)
    fmat1 = fmat1_tmp
    call cpu_time(end_time)
    ! results
    if(myid .eq. 0)then
        write(*,*)'CPU(mpi:',numprocs,') = '
        write(*,*)'  sum(fmat) = ',sum(fmat1)
        ! time
        write(*,*)'  time      = ',end_time-start_time
    endif
    
    if(myid .eq. 0)then
        call cpu_time(start_time)
        do i = 1,nvpm
            do j = 1,nvpm
                rr = 0.0
                do k = 1, ncfgs
                    do l = 1, ncfgs
                        rr = rr + phimat(i,k,l)*phimat(j,k,l)
                    enddo 
                enddo 
                fmat1(i,j) = rr
            enddo 
        enddo 
        call cpu_time(end_time)
        ! results
        if(myid .eq. 0)then
            write(*,*)'CPU(serial): '
            write(*,*)'  sum(fmat) = ',sum(fmat1)
            ! time
            write(*,*)'  time      = ',end_time-start_time
        endif
    endif
    
    !
!   istat = cudaeventrecord(startEvent,0)
    if(myid .eq. 0)then
        allocate(phimat_dev(nvpm,ncfgs,ncfgs))
        allocate(fmat_dev(nvpm,nvpm))
        istat = cudaeventrecord(startEvent,0)
        phimat_dev = phimat
        fmat_dev = 0.0
!       istat = cudaeventrecord(startEvent,0)
        !$cuf kernel do (2) <<<(*,*),(50,50)>>>
        do i = 1,nvpm
            do j = 1,nvpm
                r_tmp = 0.0
                do k = 1, ncfgs
                    do l = 1, ncfgs
                        r_tmp = r_tmp + phimat_dev(i,k,l)*phimat_dev(j,k,l)
                    enddo 
                enddo 
                fmat_dev(i,j) = r_tmp
            enddo 
        enddo 
        
!       call syncthreads()
        fmat1 = fmat_dev 
        istat = cudaeventrecord(stopEvent,0)
        istat = cudaeventsynchronize(stopEvent)
        istat = cudaeventelapsedtime(time,startEvent,stopEvent)
        
        ! results
        write(*,*)'GPU: '
        write(*,*)'  sum(fmat) = ',sum(fmat1)
        !
        ! time
        write(*,*)'  time      = ',time/1000.0
        deallocate(phimat_dev)
        deallocate(fmat_dev)
    endif 
    call MPI_BCAST(fmat1,nvpm*nvpm,MPI_DOUBLE,0,MPI_COMM_WORLD,ierr)
    call MPI_Finalize(ierr)
end program 

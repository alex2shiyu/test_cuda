program fmat
    use cudafor
    use mpi
    implicit none
    integer :: i, j, k, l, m
    integer,parameter :: nvpm  = 10
    real(8) :: start_time, end_time
    integer, device, allocatable :: phimat1(:,:), phimat2(:,:),phimat3(:,:)
    integer :: phimat1_h(nvpm,nvpm),phimat2_h(nvpm,nvpm),phimat3_h(nvpm,nvpm)
    !
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
   
    phimat1_h = 1
    phimat2_h = 2

        istat = cudaSetDevice(0)
        allocate(phimat1(nvpm,nvpm))
        allocate(phimat3(nvpm,nvpm))
        phimat1 = phimat1_h
        istat = cudaSetDevice(1)
        allocate(phimat2(nvpm,nvpm))
        phimat2 = phimat2_h
    
        istat = cudaSetDevice(0)

    ! creat time event
    istat = cudaeventcreate(startEvent) 
    istat = cudaeventcreate(stopEvent) 
    !
        phimat3 = phimat2
        !$cuf kernel do (2) <<<(*,*),(*,*)>>>
        do i = 1, nvpm
            do j = 1, nvpm
                phimat3(i,j) = phimat1(i,j) + phimat2(i,j)
            enddo
        enddo 
        phimat3_h = phimat3
        write(*,*)'sum(phimat3) = ',sum(phimat3_h)

        deallocate(phimat1)
        deallocate(phimat3)
        istat = cudaSetDevice(1)
        deallocate(phimat2)

    ! assign the phimat and fmat
    call MPI_Finalize(ierr)
end program 

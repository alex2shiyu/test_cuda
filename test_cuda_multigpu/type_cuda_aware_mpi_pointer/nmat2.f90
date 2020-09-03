program fmat
    use cudafor
    use mpi
    use mod_type
    implicit none
    integer :: i, j, k, l, m
    integer,parameter :: nvpm  = 10
    !
    !
    !
    ! mpi
    integer :: numprocs 
    integer :: myid,cnt_mpi,iorb,istat
    integer :: ierr
    integer :: namelen
    real(kind=8), allocatable :: h0(:,:), h1(:,:), h2(:,:)
    real(kind=8), device, pointer :: p1(:,:), p2(:,:)
    character*(MPI_MAX_PROCESSOR_NAME) processor_name
    type(test_type), target, allocatable :: type1(:)
    integer :: request_send, request_recv
    real(kind=8), allocatable :: mat3_d(:,:)
    integer :: status_send(mpi_status_size), status_recv(mpi_status_size)

    !mpi
    call MPI_Init(ierr)
    call MPI_COMM_RANK(mpi_comm_world, myid, ierr)
    call MPI_COMM_SIZE(mpi_comm_world, numprocs, ierr)
    call MPI_GET_PROCESSOR_NAME(processor_name,namelen,ierr)
    call MPI_BARRIER(mpi_comm_world,ierr)
    write(*,*)'mpi_init done.'
    allocate(type1(2))
    allocate(h0(nvpm,nvpm))
    allocate(h1(nvpm,nvpm))
    allocate(mat3_d(nvpm,nvpm))
    do i = 1, 2
        type1(i)%ndim = nvpm
    enddo 
    h0 = 0.d0
    h1 = 0.d0
    mat3_d = 0.d0
    do i = 1, nvpm
        h0(i,i) = 1.d0
    enddo

    if(myid .eq. 0) then
        istat = cudaSetDevice(myid+2)
        allocate(type1(myid+1)%mat1(nvpm,nvpm))
    elseif(myid .eq. 1)then
        istat = cudaSetDevice(myid+2)
        allocate(type1(myid+1)%mat1(nvpm,nvpm))
        p1 => type1(myid+1)%mat1
        p1 = h0
    endif
    !
    write(*,*)'mpi_init done. 2'
    call MPI_BARRIER(MPI_comm_world,ierr) 
    write(*,*)'mpi_init done. 2.1'
    if(myid .eq. 1)then
!       p1 => type1(myid+1)%mat1
!       call mpi_Isend(p1,nvpm*nvpm,MPI_real8,0,200,MPI_comm_world,request_send,ierr)
        call mpi_Isend(type1(myid+1)%mat1,nvpm*nvpm,MPI_real8,0,200,MPI_comm_world,request_send,ierr)
        call mpi_wait(request_send,status_send,ierr)
    endif
    write(*,*)'mpi_init done. 3, myid =',myid
    write(*,*)'mpi_init done. 4'
    if(myid .eq. 0)then
!       p2 => type1(myid+1)%mat1
!       call mpi_Irecv(p2,nvpm*nvpm,MPI_real8,1,200,MPI_comm_world,request_recv,ierr)
        call mpi_Irecv(mat3_d,nvpm*nvpm,MPI_real8,1,200,MPI_comm_world,request_recv,ierr)
        call mpi_wait(request_recv,status_recv,ierr)
        write(*,'(a,6(1x,I5),a,I3)')'status_recv',status_recv,'myid=',Myid

!       h1 = p2
        h1 = mat3_d
    endif 
        write(*,*)'sum(phimat3-2) = ',sum(h1),'myid=',myid
    write(*,*)'mpi_init done. 5'

    if(myid .eq. 0) then
        allocate(type1(myid+1)%mat1(nvpm,nvpm))
    elseif(myid .eq. 1)then
        allocate(type1(myid+1)%mat1(nvpm,nvpm))
    endif
    deallocate(type1)
    deallocate(h0)
    deallocate(h1)
    deallocate(mat3_d)

    ! assign the phimat and fmat
    call MPI_Finalize(ierr)
end program 

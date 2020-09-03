program fmat
    use cudafor
    use mpi
    implicit none
    integer :: i, j, k, l, m
    integer,parameter :: nvpm  = 10
    real(8) :: start_time, end_time
    integer, device, allocatable, target :: phimat1(:,:), phimat2(:,:),phimat2_2(:,:),phimat3(:,:),phimat3_2(:,:)
    integer, device, pointer :: phimat1_p(:,:), phimat2_p(:,:),phimat2_2_p(:,:),phimat3_p(:,:),phimat3_2_p(:,:)
    integer :: phimat1_h(nvpm,nvpm),phimat2_h(nvpm,nvpm),phimat3_h(nvpm,nvpm)
    integer :: status(mpi_status_size)
    integer :: request_send(2), request_recv(2)
    integer :: status_send(mpi_status_size,2), status_recv(mpi_status_size,2)
    !
    !
    type(cudaEvent)   :: startEvent, stopEvent
    real    :: time
    integer :: istat
    !
    ! mpi
    integer :: numprocs 
    integer :: myid,cnt_mpi,iorb
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

    if(myid .eq. 0) then
        istat = cudaSetDevice(myid+2)
        allocate(phimat1(nvpm,nvpm))
        allocate(phimat3(nvpm,nvpm))
        allocate(phimat3_2(nvpm,nvpm))
        phimat1 = phimat1_h
    elseif(myid .eq. 1)then
        istat = cudaSetDevice(myid+2)
        allocate(phimat2(nvpm,nvpm))
        allocate(phimat2_2(nvpm,nvpm))
        phimat2 = phimat2_h
        phimat2_2 = phimat2_h
    endif
    write(*,*)'mpi_init done. 1'
    ! creat time event
    istat = cudaeventcreate(startEvent) 
    istat = cudaeventcreate(stopEvent) 
    !
    write(*,*)'mpi_init done. 2'
    call MPI_BARRIER(MPI_comm_world,ierr) 
    write(*,*)'mpi_init done. 2.1'
    write(*,*)'length(mpi_status_size) = ', mpi_status_size
    if(myid .eq. 1)then
!       call mpi_send(phimat2,nvpm*nvpm,MPI_INTEGER,0,200,MPI_comm_world,ierr)
!       call mpi_send(phimat2_2,nvpm*nvpm,MPI_INTEGER,0,201,MPI_comm_world,ierr)
         phimat2_p => phimat2; phimat2_2_p => phimat2_2
        call mpi_Isend(phimat2_p,nvpm*nvpm,MPI_INTEGER,0,200,MPI_comm_world,request_send(1),ierr)
        call mpi_Isend(phimat2_2_p,nvpm*nvpm,MPI_INTEGER,0,201,MPI_comm_world,request_send(2),ierr)
!       call mpi_waitall(2,request_send,mpi_statuses_ignore,ierr)
        call mpi_waitall(2,request_send,status_send,ierr)
        do iorb = 1, 2
            write(*,'(a,I3,a,6(1x,I5),a,I3)')'status_send(',iorb,'):',status_send(:,iorb),'myid=',Myid
        enddo
    endif
    write(*,*)'mpi_init done. 3, myid =',myid
    write(*,*)'mpi_init done. 4'
    if(myid .eq. 0)then
!       phimat3 = phimat2
!       istat = cudaMemcpyPeer(phimat3,0,phimat2,1,nvpm*nvpm)
!       call mpi_recv(phimat3,nvpm*nvpm,MPI_INTEGER,1,200,MPI_comm_world,status,ierr)
!       call mpi_recv(phimat3_2,nvpm*nvpm,MPI_INTEGER,1,201,MPI_comm_world,status,ierr)
        phimat3_p => phimat3; phimat3_2_p => phimat3_2
        call mpi_Irecv(phimat3_p,nvpm*nvpm,MPI_INTEGER,1,200,MPI_comm_world,request_recv(1),ierr)
        call mpi_Irecv(phimat3_2_p,nvpm*nvpm,MPI_INTEGER,1,201,MPI_comm_world,request_recv(2),ierr)
        call mpi_waitall(2,request_recv,status_recv,ierr)
        do iorb = 1, 2
            write(*,'(a,I3,a,6(1x,I5),a,I3)')'status_recv(',iorb,'):',status_recv(:,iorb),'myid=',Myid
        enddo

        !$cuf kernel do (2) <<<(*,*),(*,*)>>>
        do i = 1, nvpm
            do j = 1, nvpm
                phimat3(i,j) = phimat1(i,j) + phimat3(i,j)! + phimat3_2(i,j)
            enddo
        enddo 
        phimat3_h = phimat3
        write(*,*)'sum(phimat3-1) = ',sum(phimat3_h)
        phimat3_h = phimat3_2 
        write(*,*)'sum(phimat3-2) = ',sum(phimat3_h)
    endif 
    write(*,*)'mpi_init done. 5'

    if(myid .eq. 0) then
        deallocate(phimat1)
        deallocate(phimat3)
        deallocate(phimat3_2)
    elseif(myid .eq. 1)then
        deallocate(phimat2)
        deallocate(phimat2_2)
    endif

    ! assign the phimat and fmat
    call MPI_Finalize(ierr)
end program 

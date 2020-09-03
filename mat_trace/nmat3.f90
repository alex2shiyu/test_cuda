program fmat
    use cudafor
    implicit none
    integer :: i, j, k, l, m
    integer,parameter :: nvpm  = 100
    real(8)           :: phimat(nvpm,nvpm,nvpm)
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

    !mpi
    ! creat time event
    istat = cudaeventcreate(startEvent) 
    istat = cudaeventcreate(stopEvent) 
    !
    ! assign the phimat and fmat
    phimat = 0.0
    fmat1  = 0.0
    do i = 1, nvpm
        do j = 1, nvpm
            phimat(i,j,j) = 1.0
        enddo
    enddo



    
        allocate(phimat_dev(nvpm,nvpm,nvpm))
        allocate(fmat_dev(nvpm,nvpm))

        phimat_dev = phimat 
!       istat = cudamemcpy(fmat_dev, phimat_dev(1,:,:), nvpm*nvpm,cudamemcpydevicetodevice)
        fmat_dev = phimat_dev(1,:,:)
        fmat1 = fmat_dev

        write(*,*)'fmat1 = ', sum(fmat1)

        deallocate(phimat_dev)
        deallocate(fmat_dev)
end program 

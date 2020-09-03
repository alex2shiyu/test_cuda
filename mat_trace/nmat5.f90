program fmat
    use cudafor
    implicit none
    integer :: i, j, k, l, m, n
    integer,parameter :: nvpm  = 400
    integer,parameter :: norb  = 14

    real(8)           :: rr
    real(8) :: start_time, end_time
    !
    real(8), device, allocatable    :: dim4mat_d(:,:,:)
    real(8), device, allocatable    :: vec1_d(:), phimat_d(:,:), vec2_d(:)

    real(8), allocatable    :: vec1(:), phimat(:,:), vec2(:)
    real(8), allocatable    :: dim4mat(:,:,:)
    !
    type(cudaEvent)   :: startEvent, stopEvent
    real    :: time
    integer :: istat
    !
    !mpi
    ! creat time event
    istat = cudaeventcreate(startEvent) 
    istat = cudaeventcreate(stopEvent) 
    !
    allocate(dim4mat(norb,norb,norb))
    allocate(vec1(nvpm))
    allocate(vec2(nvpm))
    allocate(phimat(nvpm,nvpm))
    
    ! initial
    do i = 1, nvpm
        vec1(i) = real(i)
        vec2(i) = real(i)
        do j = 1, nvpm
            phimat(i,j) = real(i)
        enddo 
    enddo 



    allocate(dim4mat_d(norb,norb,norb))
    allocate(vec1_d(nvpm))
    allocate(vec2_d(nvpm))
    allocate(phimat_d(nvpm,nvpm))

    
    istat = cudaeventrecord(startEvent,0)
    vec1_d = vec1; vec2_d = vec2; phimat_d = phimat
    !$cuf kernel do (3) <<<(*,*),(*,*)>>>
    do i = 1, norb
        do j = 1, norb
            do k = 1, norb
                    rr = 0.d0
                    do m = 1, nvpm
                        do n = 1, nvpm
                            rr = rr + vec1_d(m) * phimat_d(m,n) * vec2_d(n)
                        enddo 
                    enddo 
                    dim4mat_d(i,j,k) = rr
            enddo 
        enddo 
    enddo
    dim4mat = dim4mat_d
    istat = cudaeventrecord(stopEvent,0)
    istat = cudaeventsynchronize(stopEvent)
    istat = cudaeventelapsedtime(time,startEvent,stopEvent)
    write(*,*)'GPU(cuf kernel external loop): '
    write(*,*)'  * sum(dim4_mat_d) = ',sum(dim4mat)
    write(*,*)'  * time      = ',time/1000.0

    istat = cudaeventrecord(startEvent,0)
    vec1_d = vec1; vec2_d = vec2; phimat_d = phimat
    do i = 1, norb
        do j = 1, norb
            do k = 1, norb
                    rr = 0.d0
                    !$cuf kernel do (2) <<<(*,*),(*,*)>>>
                    do m = 1, nvpm
                        do n = 1, nvpm
                            rr = rr + vec1_d(m) * phimat_d(m,n) * vec2_d(n)
                        enddo 
                    enddo 
                    dim4mat(i,j,k) = rr
            enddo 
        enddo 
    enddo
    istat = cudaeventrecord(stopEvent,0)
    istat = cudaeventsynchronize(stopEvent)
    istat = cudaeventelapsedtime(time,startEvent,stopEvent)
    write(*,*)'GPU(cuf kernel internal loop): '
    write(*,*)'  * sum(dim4_mat) = ',sum(dim4mat)
    write(*,*)'  * time      = ',time/1000.0

    if(allocated(dim4mat_d))deallocate(dim4mat_d)
    if(allocated(vec1_d))deallocate(vec1_d)
    if(allocated(vec2_d))deallocate(vec2_d)
    if(allocated(phimat_d))deallocate(phimat_d)

    if(allocated(dim4mat))deallocate(dim4mat)
    if(allocated(vec1))deallocate(vec1)
    if(allocated(vec2))deallocate(vec2)
    if(allocated(phimat))deallocate(phimat)

    
end program 

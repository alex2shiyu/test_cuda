module mod_type
    use cudafor
    implicit none
    private

    type, public :: test_type
        integer  :: ndim
        real(kind=8), allocatable, device :: mat1(:,:) 
    end type

end module 

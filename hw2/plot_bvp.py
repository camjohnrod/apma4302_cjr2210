import numpy as np
from petsc4py import PETSc
import subprocess
import matplotlib.pyplot as plt

def read_hdf5_vec(filename, vec_name):
    """
    Read PETSc HDF5 viewer output and convert to numpy arrays.
    
    Parameters:
    filename: str - path to the HDF5 file
    vec_name: str - name of the vector to read  
    
    Returns:
    numpy array containing the data
    """
    # Create a viewer for reading HDF5 files
    viewer = PETSc.Viewer().createHDF5(filename, 'r')   
    
    # Create a Vec to load the data
    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setName(vec_name)
    vec.load(viewer)
   
    # Convert to numpy array
    array = vec.getArray()
    
    # Clean up
    vec.destroy()
    viewer.destroy()
    
    return array.copy()


def plot_bvp_solution(x, u_numeric, u_exact):
    """
    Plot the numerical and exact solutions of the BVP.
    
    Parameters:
    x: numpy array - grid points
    u_numeric: numpy array - numerical solution
    u_exact: numpy array - exact solution
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, u_numeric, 'b-', label='Numerical Solution', linewidth=2)
    ax1.plot(x, u_exact, 'r--', label='Exact Solution', linewidth=2)
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    ax1.set_title('BVP Numerical vs Exact Solution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1 .grid(True)

    ax2.plot(x, u_numeric - u_exact, 'g--', label='Error', linewidth=1)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.set_ylim(-np.max(np.abs(u_numeric - u_exact)) * 3., np.max(np.abs(u_numeric - u_exact)) * 3.)

    plt.show()

def run_bvp(m, k, gamma=0.0, c=3.0, np_procs=1):
    result = subprocess.run(
        ["./bvp",
        "-bvp_m", str(m),
        "-bvp_gamma", str(gamma),
        "-bvp_k", str(k),
        "-bvp_c", str(c),
        "-ksp_rtol", "1e-12",
        "-ksp_atol", "1e-14"],
        capture_output=True, text=True
    )
    # check both stdout and stderr
    output = result.stdout + result.stderr
    for line in output.split('\n'):
        if 'error' in line and 'xexact' in line:  # more specific match
            return float(line.split('=')[-1].strip())
    raise RuntimeError(
        f"Could not parse error.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

def plot_convergence():
    k_vec = [1, 5, 10]
    m_vec = [40, 80, 160, 320, 640, 1280]
    hs = np.array([1.0 / (m - 1) for m in m_vec])

    fig, ax = plt.subplots(figsize=(8, 6))

    for k in k_vec:
        errors = []
        for m in m_vec:
            err = run_bvp(m, k)
            errors.append(err)
            print(f"k={k}, m={m}, h={1/(m-1):.5f}, error={err:.3e}")
        errors = np.array(errors)

        slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
        print(f"k={k}: order of convergence = {slope:.2f}")

        ax.loglog(hs, errors, marker='o', label=f'k={k} (order={slope:.3f})')

    # ax.loglog(hs, hs**2 / hs[0]**2 * errors[0], 'k--', label='O(h²)')

    ax.set_xlabel('h', fontsize=14)
    ax.set_ylabel('||u - u_exact||_2', fontsize=14)
    ax.set_title('Convergence of the modified solver', fontsize=16)
    ax.legend(fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Example usage
    # read numerical and exact solutions from HDF5 files
    h5_filename = 'bvp_solution.h5'  # Update with your actual filename
    u = read_hdf5_vec(h5_filename, 'u') 
    u_exact = read_hdf5_vec(h5_filename, 'uexact')

    x = np.linspace(0, 1, len(u))  
    
    plot_bvp_solution(x, u, u_exact)
    plot_convergence()

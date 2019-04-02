#!/usr/bin/env python
# =====================================
# = File for hosting useful functions =
# = so we don't clutter the iPython   =
# = notebook.                         =
# = Joseph Murphy                     =
# = MatSci 331                        =
# = Pset 4                            =
# = Copied from pset_3 directory      =
# =====================================

# Math
import numpy as np
from numpy import fft

# Progress bar
from tqdm import tqdm

# Debugging
from pdb import set_trace

# ==========================================================================
# ================ Functions below are new to problem set 4 ================
# ==========================================================================

# 1.1
def heat_capacity(kb_T_time_series, num_atoms):
    '''
    Calculate the heat capacity per atom using equation 2 in the problem set statement.
    Returned in units of kb.

    '''
    # kb = 1.38064852e-23 # m^2 kg s^-2 K^-1

    mean = np.mean(kb_T_time_series)
    mean_sqr = mean * mean
    var = np.var(kb_T_time_series)

    c_v = (3 * mean_sqr) / (2 * mean_sqr - 3 * var * num_atoms)

    return c_v # Result is returned in units of kb


def heat_capacity_MC(e_pot_time_series, kb_T, num_atoms):
    '''
    Calculate the heat capacity per atom using equation 4 in the problem set statement.

    Returned in units of kb.
    '''

    var = np.var(e_pot_time_series)

    c_v = (var / (num_atoms * kb_T * kb_T)) + 1.5

    return c_v

def delta_energy_single_atom_disp(atoms, latvec_mat, r_c, disp_atom_index, disp_vec, sigma = 1.0, epsilon = 1.0):
    '''
    Used in MC_MD_loop_fast(). Calculates the change in the potential energy of
    the crystal due to the displacement of a single atom. No loops.
    '''

    num_atoms = atoms.shape[0]

    # The new coordinates of the atom that was displaced
    disp_atom_coords = atoms[disp_atom_index]

    # The old coordinates of the atom (before it was displaced)
    old_atom_coords  = disp_atom_coords - disp_vec

    # These are constants so calculate them out here.
    r_c_sqr = r_c * r_c
    r_c_6   = r_c_sqr * r_c_sqr * r_c_sqr
    r_c_12  = r_c_6 * r_c_6

    # Sigma will usually be 1.0, so this isn't totally necessary, but including
    # the functionality in case it's needed later.
    sigma_sqr = sigma*sigma
    sigma_6   = sigma_sqr*sigma_sqr*sigma_sqr
    sigma_12  = sigma_6*sigma_6

    # e_cut is the second term in the LJ potential (constant, so calculated out here)
    e_cut = -sigma_6/r_c_6 + sigma_12/r_c_12

    # ----------------------------------------
    # Find the displacements in the (displaced crystal) from the displaced atom
    disps_atoms_t = np.copy(atoms) - atoms[disp_atom_index]
    shifts = np.hstack((np.around(disps_atoms_t[:, 0]/latvec_mat[0,0]).reshape(-1, 1),
    np.around(disps_atoms_t[:, 1]/latvec_mat[1,1]).reshape(-1, 1),
    np.around(disps_atoms_t[:, 2]/latvec_mat[2,2]).reshape(-1, 1)))

    disps_atoms_t -= np.matmul(shifts, latvec_mat) # shape: (N, 3) = (N, 3) x (3, 3)
    disps_atoms_t_sqr = np.square(np.linalg.norm(disps_atoms_t, axis=1)) # shape: (N, 1)

    # Indices of the nearest neighbors (but don't count self-interaction)
    atoms_nn_inds_t = np.nonzero(disps_atoms_t_sqr < r_c_sqr)[0]
    self_ind = np.argwhere(atoms_nn_inds_t == disp_atom_index)
    atoms_nn_inds_t = np.delete(atoms_nn_inds_t, self_ind)

    # Now just need to calculate the energy contribution from each of the nearest neighbors
    disps_atoms_nn_t_sqr = disps_atoms_t_sqr[atoms_nn_inds_t] # Shape: (atoms_nn_inds_t.shape[0], 1)
    inv_r_6 = np.reciprocal(np.power(disps_atoms_nn_t_sqr, 3))
    inv_r_12 = np.square(inv_r_6)
    e_disp = -sigma_6 * inv_r_6 + sigma_12*inv_r_12 # Shape: (atoms_nn_inds_t.shape[0], 1)

    e_new = e_disp - e_cut # Shape: (atoms_nn_inds_t.shape[0], 1)

    e_new = 4*np.sum(e_new) # scalar
    # ----------------------------------------

    # Now do the same thing for the old atom config (before the atom atoms[j] was displaced)
    # ----------------------------------------
    atoms_t_minus_1 = np.copy(atoms)
    atoms_t_minus_1[disp_atom_index] -= disp_vec
    disps_atoms_t_minus_1 = atoms_t_minus_1 - atoms_t_minus_1[disp_atom_index]
    shifts_t_minus_1 = np.hstack((np.around(disps_atoms_t_minus_1[:, 0]/latvec_mat[0,0]).reshape(-1, 1),
    np.around(disps_atoms_t_minus_1[:, 1]/latvec_mat[1,1]).reshape(-1, 1),
    np.around(disps_atoms_t_minus_1[:, 2]/latvec_mat[2,2]).reshape(-1, 1)))

    disps_atoms_t_minus_1 -= np.matmul(shifts_t_minus_1, latvec_mat) # shape: (N, 3) = (N, 3) x (3, 3)
    disps_atoms_t_minus_1_sqr = np.square(np.linalg.norm(disps_atoms_t_minus_1, axis=1)) # shape: (N, 1)

    # Indices of the nearest neighbors
    atoms_nn_inds_t_minus_1 = np.nonzero(disps_atoms_t_minus_1_sqr < r_c_sqr)[0]
    self_ind_t_minus_1 = np.argwhere(atoms_nn_inds_t_minus_1 == disp_atom_index)
    atoms_nn_inds_t_minus_1 = np.delete(atoms_nn_inds_t_minus_1, self_ind_t_minus_1)

    # Now just need to calculate the energy contribution from each of the nearest neighbors
    disps_atoms_nn_t_minus_1_sqr = disps_atoms_t_minus_1_sqr[atoms_nn_inds_t_minus_1] # Shape: (atoms_nn_inds_t.shape[0], 1)
    inv_r_6 = np.reciprocal(np.power(disps_atoms_nn_t_minus_1_sqr, 3))
    inv_r_12 = np.square(inv_r_6)
    e_disp_t_minus_1 = -sigma_6 * inv_r_6 + sigma_12*inv_r_12 # Shape: (atoms_nn_inds_t.shape[0], 1)

    e_t_minus_1 = e_disp_t_minus_1 - e_cut # Shape: (atoms_nn_inds_t.shape[0], 1)

    e_t_minus_1 = 4*np.sum(e_t_minus_1) # scalar
    # ----------------------------------------

    e_delta = e_new - e_t_minus_1

    return e_delta

# ==========================================================================
# ================ Functions above are new to problem set 4 ================
# ==========================================================================


# ==========================================================================
# ==== Functions below copied from util_functions.py from problem set 3 ====
# ==========================================================================

def setup_cell(L, M, N, a = 2**(2.0/3.0)):
    '''
    Set up the fcc computational cell. Returns matrix of atom positions of shape (4*L*M*N, 3).
    Based on the matlab starter code released with problem set 2.

    Params
    ===========
    L, M, N (ints): The dimensions of the computational cell.
    a (float): The lattice constant.

    Returns
    ===========
    atoms (numpy array, shape = (4*L*M*N, 3)): Row corresponds to a single atom,
        columns correspond to its coordinates in real space.
    latvec_mat (numpy array, shape = (3,3)): Columns (or rows since diag) are the
        lattice vectors. Convention: 1st column is L, 2nd is M, 3rd is N.
    '''

    # Define the primitive cell
    num_atoms = 0
    basis = np.empty((4, 3))
    basis[0, :] = np.array([0.0, 0.0, 0.0])
    basis[1, :] = np.array([0.5, 0.5, 0.0])
    basis[2, :] = np.array([0.0, 0.5, 0.5])
    basis[3, :] = np.array([0.5, 0.0, 0.5])
    num_basis = basis.shape[0]

    # Matrix of atom positions
    atoms = np.empty((4*L*M*N, 3))

    # Make periodic copies of the primitive cell
    for l in range(L):
        for m in range(M):
            for n in range(N):
                for k in range(num_basis):
                    atoms[num_atoms + k, :] = basis[k, :] + np.array([l, m, n])

                num_atoms += num_basis
    # Matrix whose columns are the lattice vectors
    latvec_mat = np.array([[a*L, 0, 0], [0, a*M, 0], [0, 0, a*N]])

    # Multiply by a to get real space coordinates for the atoms
    atoms = atoms * a

    return atoms, latvec_mat

def calc_energy_and_forces(atoms, latvec_mat, r_c, sigma = 1.0, epsilon = 1.0, calc_forces = False):
    '''
    Updated E_tot_forces() function for pset_3. (use this function)

    Calculates the total energy and forces of the computational cell.

    Based on the calc_energy_faster.m file released as part of the Pset 3 distribution.

    Note from starter code:
    % Important:
    % This routine requires 2*rcut < computational cell lengths
    '''

    # The "\" continues the line since it is long.
    assert 2*r_c < latvec_mat[0,0] and 2*r_c < latvec_mat[1,1] and 2*r_c < latvec_mat[2,2], \
    "See function documentation. Need larger computationl cell size."

    # Number of atoms = number of rows in atoms matrix.
    num_atoms = atoms.shape[0]

    # These are constants so calculate them out here.
    r_c_sqr = r_c * r_c # faster than r_c**2
    r_c_6   = r_c_sqr * r_c_sqr * r_c_sqr
    r_c_12  = r_c_6 * r_c_6

    # Sigma will usually be 1.0, so this isn't totally necessary, but including
    # the functionality in case it's needed later.
    sigma_sqr = sigma*sigma
    sigma_6   = sigma_sqr*sigma_sqr*sigma_sqr
    sigma_12  = sigma_6*sigma_6

    # e_cut is the second term in the LJ potential (constant, so calculated out here)
    e_cut = -sigma_6/r_c_6 + sigma_12/r_c_12

    # Will be returning these in the end (force_mat only if calc_forces = True)
    e_tot = 0.0
    force_mat = np.zeros(atoms.shape)

    # num_atoms - 1 so we don't double count the last atom.
    for i in range(num_atoms - 1):#for i in tqdm(range(num_atoms - 1)): # tqdm progress bar.
        for j in range(i+1, num_atoms):

            disp  = atoms[i, :] - atoms[j, :]
            shift = np.array([round(disp[0]/latvec_mat[0,0]),
            round(disp[1]/latvec_mat[1,1]), round(disp[2]/latvec_mat[2,2])])

            # If the simulation becomes unstable and we lose numerical integrity.
            if np.isnan(shift).any():
                print("MD simulation unstable, exiting calcl_energy_and_forces()...")
                # Return
                e_tot *= 4*epsilon
                if calc_forces:
                    return e_tot, force_mat
                return e_tot, None

            disp -= np.matmul(shift, latvec_mat)
            d_sqr = np.dot(disp, disp)

            # Only calculate energy for atoms within cutoff radius
            # Don't include self interactions
            if (d_sqr < r_c_sqr and d_sqr > 0.0):

                # First term in the LJ potential
                inv_r_6 = 1/(d_sqr*d_sqr*d_sqr)
                inv_r_12 = inv_r_6*inv_r_6
                e_disp = -sigma_6*inv_r_6 + sigma_12*inv_r_12

                # Add the energy for this atom to the total
                e_tot += e_disp - e_cut

                # If we were told to also calculate the forces, then do that also.
                if calc_forces:
                    fac = 24*(2*sigma_12*inv_r_12 - sigma_6*inv_r_6)/d_sqr
                    force_mat[i, :] += fac * disp
                    force_mat[j, :] -= fac * disp

    e_tot *= 4*epsilon
    if calc_forces:
        return e_tot, force_mat

    # If calc_forces = False, return None for the force matrix so we don't get confused.
    return e_tot, None

# problem set 3
def initialize_velocities(atoms, latvec_mat, kb_T, dt):
    '''
    Initialize the atoms with random velocities at temperature T.

    Params
    ===========
    atoms (numpy array, shape = (4*L*M*N, 3)): Row corresponds to a single atom,
        columns correspond to its position coordinates in real space.
    latvec_mat (numpy array, shape = (3,3)): Columns (or rows since diag) are the
        lattice vectors. Convention: 1st column is L, 2nd is M, 3rd is N.

    Returns
    ===========
    velocities (numpy array, shape = (4*L*M*N, 3)): Row corresponds to a single atom,
        columns correspond to its velocity components in real space.
    atoms_old (numpy array, shape = (4*L*M*N, 3)): Atom positions minus one time
        step given the velocities just created. Row corresponds to a single atom,
        columns correspond to its coordinates in real space.
    '''
    num_atoms = atoms.shape[0]

    velocities = np.random.uniform(-1.0, 1.0, size=atoms.shape)

    # Normalize the velocities by the center of mass velocity
    vel_cm = np.sum(velocities, axis=0) / num_atoms
    velocities -= vel_cm

    kinetic_energy = 0.5 * np.sum(np.square(velocities))

    # By equipartition theorem:
    target_kinetic_energy = 1.5 * num_atoms * kb_T

    scale_factor = np.sqrt(target_kinetic_energy / kinetic_energy)
    velocities *= scale_factor

    # Backward time step for Verlet (don't end up really needing this, though)
    atoms_old = atoms - velocities * dt

    return velocities, atoms_old


# 2.1 (problem set 3)
def verlet_step(atoms, latvec_mat, r_c, forces, velocities, dt, mass = 1.0):
    '''
    Takes in r(t), f(t), and v(t), returns all matrices advanced in time by
        step dt (as well as the new energy).
    '''

    atoms_t_plus_dt = atoms + velocities * dt + 0.5 * (1/mass) * forces * dt * dt

    e_pot_t_plus_dt, forces_t_plus_dt = calc_energy_and_forces(atoms_t_plus_dt,
    latvec_mat, r_c, calc_forces=True)

    velocities_t_plus_dt = velocities + (1/ (2*mass)) * (forces + forces_t_plus_dt) * dt

    return atoms_t_plus_dt, forces_t_plus_dt, velocities_t_plus_dt, e_pot_t_plus_dt

# 2.2 (problem set 3)
def calc_KE_and_kb_T(velocities, mass = 1.0):
    '''
    Calculates the kinetic energy and the temperature, assuming equipartition,
        at time t.
    '''
    num_atoms = velocities.shape[0]

    kinetic_energy = 0.5 * mass * np.sum(np.diagonal(np.matmul(velocities, velocities.T)))

    kb_T_t = (2/(3*num_atoms)) * kinetic_energy

    return kinetic_energy, kb_T_t

# 3.2 (problem set 3)
def velocity_autocorrelation(velocity_time_series):
    '''
    Velocity time series is a 3-dimensional matrix of shape (num_atoms, 3, num_steps + 1)
    i.e. time is along the z axis.

    Computes the velocity autocorrelation function using np.fft.fft() with
        normalization of norm='ortho', effectively normalizing the FT output by
        1/sqrt{num_steps + 1}.

    '''
    num_atoms = velocity_time_series.shape[0]

    # Velocity autocorrelation
    velocity_time_series *= np.repeat(np.expand_dims(velocity_time_series[:, :, 0],
    axis=2), atoms_t.shape[2], axis=2)

    # Take fourier transform
    return (1/(3*num_atoms))*np.sum(np.sum(np.square(np.absolute(fft.fft(
    velocity_time_series, axis=2, norm='ortho'))), axis=1), axis=0)

# 6.1 (problem set 3)
def mean_squared_displacement(position_time_series):

    num_atoms = position_time_series.shape[0]

    atoms_0 = position_time_series[:, :, 0]
    atoms_t = position_time_series[:, :, 1:]
    assert len(atoms_0.shape) == 2 and len(atoms_t.shape) == 3, "Dimensions off."
    atoms_disp = atoms_t - np.repeat(np.expand_dims(atoms_0, axis=2), atoms_t.shape[2], axis=2)

    return (1/num_atoms) * np.sum(np.square(np.linalg.norm(atoms_disp, axis=1)), axis=0)

# ==========================================================================
# ==== Functions above copied from util_functions.py from problem set 3 ====
# ==========================================================================

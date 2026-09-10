import h5py
import numpy as np
import matplotlib.pyplot as plt


class SFHCatalog:
    """
    Interface de lecture des catalogues SFH stockés en HDF5.

    Datasets attendus :

    object_id
    phz_pp_median_redshift
    phz_pp_median_stellarmass
    sersic_sersic_vis_index
    age
    sfh
    tf
    t25
    t50
    t75
    t90
    t95
    t99
    slope
    in_cluster
    """

    _derived_quantity_names = ("tf", "t25", "t50", "t75", "t90", "t95", "t99", "slope")

    def __init__(self, filename):

        self.filename = filename
        self.h5 = h5py.File(filename, "r")

        # Métadonnées chargées en RAM
        self.ids = self.h5["object_id"][:]
        self.z = self.h5["phz_pp_median_redshift"][:]
        self.mass = self.h5["phz_pp_median_stellarmass"][:]
        self.sersic = self.h5["sersic_sersic_vis_index"][:]

        self.age = self.h5["age"][:]
        #in_cluster colonne si elle existe, sinon None
        self.in_cluster = self.h5["in_cluster"][:] if "in_cluster" in self.h5 else None

        # Dataset HDF5 (pas chargé en RAM)
        self.sfh = self.h5["sfh"]

        # Quantités dérivées chargées à la demande
        self._derived_quantity_cache = {}
        self.derived_quantities = {
            name: None
            for name in self._derived_quantity_names
        }

        self.id_to_index = {
            gid: i
            for i, gid in enumerate(self.ids)
        }

    # --------------------------------------------------
    # context manager
    # --------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.h5.close()

    # --------------------------------------------------
    # properties
    # --------------------------------------------------

    @property
    def n_gal(self):
        return len(self.ids)

    def _get_derived_quantity(self, name):

        if name in self._derived_quantity_cache:
            return self._derived_quantity_cache[name]

        if name not in self.h5:
            return None

        value = self.h5[name][:]
        self._derived_quantity_cache[name] = value
        self.derived_quantities[name] = value

        return value

    @property
    def tf(self):
        return self._get_derived_quantity("tf")

    @property
    def t25(self):
        return self._get_derived_quantity("t25")

    @property
    def t50(self):
        return self._get_derived_quantity("t50")

    @property
    def t75(self):
        return self._get_derived_quantity("t75")

    @property
    def t90(self):
        return self._get_derived_quantity("t90")

    @property
    def t95(self):
        return self._get_derived_quantity("t95")

    @property
    def t99(self):
        return self._get_derived_quantity("t99")

    @property
    def slope(self):
        return self._get_derived_quantity("slope")

    # --------------------------------------------------
    # galaxy selection
    # --------------------------------------------------

    def select_galaxies(
        self,
        z_range=None,
        mass_range=None,
        sersic_range=None,
        gal_type=None,
        sersic_threshold=2.0,
        in_cluster=None
    ):
        """
        Retourne les indices des galaxies sélectionnées.
        """

        mask = np.ones(self.n_gal, dtype=bool)

        if z_range is not None:
            mask &= (
                (self.z >= z_range[0])
                &
                (self.z < z_range[1])
            )

        if mass_range is not None:
            mask &= (
                (self.mass >= mass_range[0])
                &
                (self.mass < mass_range[1])
            )

        if sersic_range is not None:
            mask &= (
                (self.sersic >= sersic_range[0])
                &
                (self.sersic < sersic_range[1])
            )

        if gal_type == "late":
            mask &= self.sersic < sersic_threshold

        elif gal_type == "early":
            mask &= self.sersic > sersic_threshold

        elif gal_type in [None, "all"]:
            pass

        else:
            raise ValueError(
                "gal_type doit être "
                "'early', 'late', 'all' ou None"
            )
        
        if in_cluster is not None:

            if self.in_cluster is None:
                raise ValueError(
                    "Le catalogue ne contient pas la colonne 'in_cluster'"
                )

            if in_cluster == True:
                mask &= self.in_cluster
            elif in_cluster == False:
                mask &= ~self.in_cluster
            else:
                raise ValueError(
                    "in_cluster doit être True, False ou None"
                )

        return np.where(mask)[0]

    # --------------------------------------------------
    # access to a single galaxy
    # --------------------------------------------------

    def get_sfh(self, object_id):
        """
        Retourne les 50 réalisations de SFH
        d'une galaxie donnée.
        """

        if object_id not in self.id_to_index:
            raise KeyError(
                f"object_id={object_id} absent du catalogue"
            )

        return self.sfh[
            self.id_to_index[object_id]
        ]

    # --------------------------------------------------
    # SFH stacking
    # --------------------------------------------------

    def stack_sfh(
        self,
        idx,
        percentiles=(16, 50, 84),
        combine_realizations=True,
    ):
        """
        Compute SFH percentiles for a set of galaxies.

        Parameters:
        ----------
        idx: list of int or int (for one galaxy)
            index of the galaxy (!= object_id)
        percentiles: tuple of floats
            list of percentiles
        combine_realizations: bool
            if True, stack all SFHs and compute the percentiles on the stacked table. 
            if False, compute the percentiles for each galaxy.
        
        Returns:
        --------
        dict:
            dictionnary containing the lookbacktimes, number of galaxies and the SFH percentiles
        """

        if type(idx) is int:
            idx = [idx]

        n_gal = len(idx)

        if n_gal == 0:
            return None

        sfh = self.sfh[idx]

        # sfh shape:
        # (Ngal,50,250)

        if combine_realizations:

            sfh = sfh.reshape(-1, sfh.shape[-1]) # stack the SFHs: (n_gal, 50, 250) --> (n_gal*50, 250)

            q = np.percentile(sfh, percentiles, axis=0) # compute the percentiles on the stacked table -> (len(percentiles), 250)
        else:

            q = np.percentile(sfh, percentiles, axis=1) #compute the percentiles for each galaxy -> (len(percentiles), n_gal, 250)
            if n_gal == 1:
                q = q.reshape(-1, 250)

        return {
            "age": self.age,
            "n_gal": n_gal,
            "p16": q[0],
            "median": q[1],
            "p84": q[2],
        }

    # --------------------------------------------------
    # SFH stacking
    # --------------------------------------------------

    def plot_sfh_gal(self, idx, ax, color, label=None):
        """
        Plot the SFH for one galaxy (median of the 50 realisations and the 68% interval).
        
        Parameters:
        -----------
        idx: int
            index of the galaxy (!= object_id)
        ax: matplotlib.axes.Axes
            ax on which to plot the SFH
        color: str
            color of the line
        label: str
            label for the legend
        
        Returns:
        --------
        None
        """

        sfh_percentiles = self.stack_sfh(idx, combine_realizations=False)

        lookback_times = sfh_percentiles["age"]
        n_gal = sfh_percentiles["n_gal"]
        
        p16 = sfh_percentiles["p16"]
        median = sfh_percentiles["median"]
        p84 = sfh_percentiles["p84"]

        ax.plot(lookback_times, median/50e6, linestyle='-', label = label, color=color)
        ax.fill_between(lookback_times, p16/50e6, p84/50e6, alpha=0.2, color=color)


        return None

        

    # --------------------------------------------------
    # convenience wrapper
    # --------------------------------------------------

    def median_SFH(
        self,
        z_start,
        z_stop,
        M_start,
        M_stop,
        gal_type="all",
        sersic_cut=2.0,
        combine_realizations=True,
    ):
        """
        Compute median SFH for a given selection of galaxies.
        """

        idx = self.select_galaxies(
            z_range=(z_start, z_stop),
            mass_range=(M_start, M_stop),
            gal_type=gal_type,
            sersic_threshold=sersic_cut,
        )

        result = self.stack_sfh(
            idx,
            combine_realizations=combine_realizations
        )

        if result is None:
            return None, None, None, None, 0

        return (
            result["age"],
            result["median"],
            result["p16"],
            result["p84"],
            result["n_gal"],
        )

    # --------------------------------------------------
    # batch iterator
    # --------------------------------------------------

    def iter_sfh(
        self,
        idx,
        batch_size=1000
    ):
        """
        Itérateur utile pour calculer
        slopes, t50, tf, etc.
        """

        for start in range(
            0,
            len(idx),
            batch_size
        ):

            stop = min(
                start + batch_size,
                len(idx)
            )

            yield self.sfh[idx[start:stop]]
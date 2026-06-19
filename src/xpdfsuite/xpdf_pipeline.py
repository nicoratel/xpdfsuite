"""
xpdf_pipeline.py – Fonctions haut niveau pour l'extraction de la xPDF.

Deux points d'entrée sont fournis :

- :func:`extract_xpdf_from_h5`    – charge un fichier HDF5/NXS, moyenne les
  frames, intègre azimutalement avec pyFAI et calcule G(r).
- :func:`extract_xpdf_from_image` – prend directement un tableau numpy 2-D,
  intègre avec pyFAI et calcule G(r).
"""

import os
import numpy as np
from pyFAI import load as pyfai_load
import fabio

from .filereader import load_h5_data
from .pdf_extraction import compute_xPDF


# ---------------------------------------------------------------------------
# Fonctions internes (non exportées)
# ---------------------------------------------------------------------------

def _load_mask(mask):
    """Retourne un tableau booléen à partir d'un chemin, d'un ndarray ou de None."""
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return mask.astype(bool)
    return fabio.open(mask).data.astype(bool)


def _load_ref_image(ref, frame='mean', verbose=False):
    """Charge une image de référence 2-D à partir d'un chemin de fichier ou d'un ndarray.

    Parameters
    ----------
    ref : str ou ndarray
        Chemin vers un fichier HDF5/NXS (str) ou image 2-D déjà chargée (ndarray).
    frame : int ou 'mean'
        Utilisé uniquement si *ref* est un fichier HDF5 (voir :func:`load_h5_data`).
    verbose : bool
        Afficher les métadonnées lors du chargement HDF5.

    Returns
    -------
    ndarray
        Image 2-D de référence (float).

    Raises
    ------
    TypeError
        Si *ref* n'est ni une chaîne ni un ndarray.
    """
    if isinstance(ref, np.ndarray):
        return ref.astype(float)
    if isinstance(ref, str):
        _, img = load_h5_data(ref, frame=frame, verbose=verbose)
        return img
    raise TypeError(
        f"'ref' doit être un chemin de fichier (str) ou un ndarray, pas {type(ref).__name__}."
    )


def _load_ai_from_poni(poni_file):
    """Charge un AzimuthalIntegrator depuis un fichier .poni.

    Tente d'abord ``pyFAI.load``.  En cas d'échec (ex. ``poni_version: 2.1``
    non reconnu par les versions anciennes de pyFAI), parse les paramètres
    manuellement et construit l'intégrateur à la main.

    Parameters
    ----------
    poni_file : str
        Chemin vers le fichier .poni.

    Returns
    -------
    ai : pyFAI.AzimuthalIntegrator
    """
    try:
        return pyfai_load(poni_file)
    except (ValueError, Exception):
        pass

    # --- Parseur de secours ---
    import pyFAI
    from pyFAI.detectors import detector_factory

    params = {}
    det_name = None
    det_config = {}

    with open(poni_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            key, _, val = line.partition(':')
            key, val = key.strip(), val.strip()
            if key == 'Detector':
                det_name = val
            elif key == 'Detector_config':
                import json as _json
                try:
                    det_config = _json.loads(val)
                except Exception:
                    pass
            else:
                try:
                    params[key] = float(val)
                except ValueError:
                    pass

    # Construire le détecteur
    # Certains kwargs (ex. 'orientation', 'sensor') ne sont pas supportés par les
    # anciennes versions de pyFAI : on les retire progressivement jusqu'à succès.
    _known_extras = ['orientation', 'sensor']
    if det_name is not None:
        _config = dict(det_config)
        detector = None
        while detector is None:
            try:
                detector = detector_factory(det_name, _config)
            except Exception:
                removed = False
                for k in _known_extras:
                    if k in _config:
                        _config.pop(k)
                        removed = True
                        break
                if not removed:
                    detector = pyFAI.detectors.Detector(
                        pixel1=det_config.get('pixel1', 75e-6),
                        pixel2=det_config.get('pixel2', 75e-6),
                    )
    else:
        detector = pyFAI.detectors.Detector(
            pixel1=det_config.get('pixel1', 75e-6),
            pixel2=det_config.get('pixel2', 75e-6),
        )

    ai = pyFAI.AzimuthalIntegrator(detector=detector)
    ai.setFit2D(
        directDist=params.get('Distance', 1.0) * 1e3,  # m → mm
        centerX=params.get('Poni2', 0.0) / detector.pixel2,
        centerY=params.get('Poni1', 0.0) / detector.pixel1,
        tilt=0.0,
        tiltPlanRotation=0.0,
    )
    ai.rot1 = params.get('Rot1', 0.0)
    ai.rot2 = params.get('Rot2', 0.0)
    ai.rot3 = params.get('Rot3', 0.0)
    ai.wavelength = params.get('Wavelength', 1e-10)
    return ai


def _integrate_image(image, poni_file, mask=None, npt=2500):
    """Intégration azimutale d'une image 2-D avec pyFAI.

    Parameters
    ----------
    image : ndarray
        Image 2-D (en counts).
    poni_file : str
        Chemin vers le fichier de calibration pyFAI (.poni).
    mask : str, ndarray ou None
        Masque : chemin vers un fichier fabio, tableau booléen, ou None.
    npt : int
        Nombre de points dans le profil 1-D résultant.

    Returns
    -------
    q : ndarray
        Vecteur de diffusion en Å⁻¹.
    I : ndarray
        Intensité intégrée azimutalement.
    """
    ai = _load_ai_from_poni(poni_file)
    mask_arr = _load_mask(mask)
    q, I = ai.integrate1d(
        image, npt,
        mask=mask_arr,
        unit="q_A^-1",
        polarization_factor=0.99,
    )
    return q, I


def _save_gr(r, G, outputfile, source_file, composition,
             bgscale, qmin, qmax, qmaxinst, rmin, rmax, rstep, rpoly):
    """Écrit G(r) dans un fichier .gr avec un en-tête compatible PDFgetX3."""
    header = '[DEFAULT]\n\nversion = xpdfsuite 1.0\n\n'
    header += '# PDF calculation setup\n'
    header += 'dataformat = q_A\n'
    header += f'inputfile = {source_file or "ndarray"}\n'
    header += 'outputtype = gr\n'
    header += 'mode = xrays\n'
    header += f'composition = {composition}\n'
    header += f'bgscale = {bgscale:.2f}\n'
    header += f'rpoly = {rpoly:.2f}\n'
    header += f'qmaxinst = {qmaxinst if qmaxinst is not None else qmax:.2f}\n'
    header += f'qmin = {qmin:.2f}\n'
    header += f'qmax = {qmax:.2f}\n'
    header += f'rmin = {rmin:.2f}\n'
    header += f'rmax = {rmax:.2f}\n'
    header += f'rstep = {rstep:.4f}\n\n'
    header += '#### start data\n'
    header += '#L r(Å)  G(Å^-2)\n'

    with open(outputfile, 'w') as f:
        f.write(header)
        for ri, Gi in zip(r, G):
            f.write(f'{ri:.4f}  {Gi:.6f}\n')

    print(f'PDF sauvegardée dans : {outputfile}')


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def extract_xpdf_from_h5(
    h5_file,
    poni_file,
    composition,
    mask=None,
    npt=2500,
    frame='mean',
    ref=None,
    bgscale=1.0,
    qmin=1.5,
    qmax=24.0,
    qmaxinst=None,
    rmin=0.0,
    rmax=50.0,
    rstep=0.01,
    rpoly=1.4,
    Lorch=True,
    outputfile=None,
    plot=False,
    verbose=True,
):
    """
    Extraire la fonction de distribution de paires G(r) à partir d'un fichier HDF5/NXS.

    Charge le stack d'images, effectue la moyenne (ou sélectionne un frame),
    intègre azimutalement avec pyFAI, puis appelle
    :func:`~xpdfsuite.pdf_extraction.compute_xPDF`.

    Parameters
    ----------
    h5_file : str
        Chemin vers le fichier HDF5 / NXS.
    poni_file : str
        Chemin vers le fichier de calibration pyFAI (.poni).
    composition : str
        Formule chimique de l'échantillon (ex. ``'CeO2'``, ``'Au'``).
    mask : str ou ndarray, optional
        Chemin vers un fichier masque fabio ou tableau booléen
        (``True`` = pixel masqué). Par défaut ``None``.
    npt : int, optional
        Nombre de points pour l'intégration azimutale. Par défaut 2500.
    frame : int ou 'mean', optional
        Indice du frame à utiliser, ou ``'mean'`` (défaut) pour moyenner
        tous les frames.
    ref : str ou ndarray, optional
        Signal de référence (fond) à soustraire. Peut être :

        - un chemin vers un fichier HDF5/NXS (str) — les frames sont
          moyennés selon le même paramètre ``frame`` ;
        - une image 2-D numpy déjà chargée (ndarray).

        Si ``None`` (défaut), aucune soustraction n'est effectuée.
    bgscale : float, optional
        Facteur d'échelle appliqué à la référence. Par défaut 1.0.
    qmin : float, optional
        Limite inférieure de Q pour la transformée de Fourier (Å⁻¹).
        Par défaut 1.5.
    qmax : float, optional
        Limite supérieure de Q pour la transformée de Fourier (Å⁻¹).
        Par défaut 24.0.
    qmaxinst : float, optional
        Limite supérieure de Q pour le fit du fond polynomial. Par défaut
        égal à ``qmax``.
    rmin, rmax, rstep : float, optional
        Étendue et pas de l'axe r (Å). Par défaut 0.0, 50.0, 0.01.
    rpoly : float, optional
        Paramètre de contrôle du degré polynomial (convention PDFgetX3).
        Par défaut 1.4.
    Lorch : bool, optional
        Appliquer la fonction de modification de Lorch. Par défaut ``True``.
    outputfile : str, optional
        Chemin du fichier .gr de sortie. Si ``None``, aucun fichier n'est
        écrit.
    plot : bool, optional
        Afficher les graphes de diagnostic. Par défaut ``False``.
    verbose : bool, optional
        Afficher les métadonnées lues dans le fichier HDF5. Par défaut
        ``True``.

    Returns
    -------
    r : ndarray
        Axe de distance réelle en Å.
    G : ndarray
        Fonction de distribution de paires réduite G(r) en Å⁻².
    """
    # Chargement et moyennage de l'image
    _, image = load_h5_data(h5_file, frame=frame, verbose=verbose)

    # Intégration azimutale de l'échantillon
    q, I = _integrate_image(image, poni_file, mask=mask, npt=npt)

    # Intégration azimutale de la référence (fond) si fournie
    I_ref = None
    if ref is not None:
        ref_image = _load_ref_image(ref, frame=frame, verbose=verbose)
        _, I_ref = _integrate_image(ref_image, poni_file, mask=mask, npt=npt)

    # Calcul de la PDF
    r, G = compute_xPDF(
        q, I,
        composition,
        Iref=I_ref,
        bgscale=bgscale,
        qmin=qmin,
        qmax=qmax,
        qmaxinst=qmaxinst,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        rpoly=rpoly,
        Lorch=Lorch,
        plot=plot,
    )

    if outputfile is not None:
        _save_gr(r, G, outputfile, h5_file, composition,
                 bgscale, qmin, qmax, qmaxinst, rmin, rmax, rstep, rpoly)

    return r, G


def extract_xpdf_from_image(
    image,
    poni_file,
    composition,
    mask=None,
    npt=2500,
    ref=None,
    bgscale=1.0,
    qmin=1.5,
    qmax=24.0,
    qmaxinst=None,
    rmin=0.0,
    rmax=50.0,
    rstep=0.01,
    rpoly=1.4,
    Lorch=True,
    outputfile=None,
    plot=False,
):
    """
    Extraire la fonction de distribution de paires G(r) à partir d'une image numpy 2-D.

    Intègre azimutalement l'image avec pyFAI, puis appelle
    :func:`~xpdfsuite.pdf_extraction.compute_xPDF`.

    Parameters
    ----------
    image : ndarray
        Image 2-D (en counts). Peut être un frame unique ou une image
        déjà moyennée.
    poni_file : str
        Chemin vers le fichier de calibration pyFAI (.poni).
    composition : str
        Formule chimique de l'échantillon (ex. ``'CeO2'``, ``'Au'``).
    mask : str ou ndarray, optional
        Chemin vers un fichier masque fabio ou tableau booléen
        (``True`` = pixel masqué). Par défaut ``None``.
    npt : int, optional
        Nombre de points pour l'intégration azimutale. Par défaut 2500.
    ref : str ou ndarray, optional
        Signal de référence (fond) à soustraire. Peut être :

        - une image 2-D numpy (ndarray) ;
        - un chemin vers un fichier HDF5/NXS (str) — les frames seront
          moyennés (``frame='mean'``).

        Si ``None`` (défaut), aucune soustraction n'est effectuée.
    bgscale : float, optional
        Facteur d'échelle appliqué à la référence. Par défaut 1.0.
    qmin : float, optional
        Limite inférieure de Q pour la transformée de Fourier (Å⁻¹).
        Par défaut 1.5.
    qmax : float, optional
        Limite supérieure de Q pour la transformée de Fourier (Å⁻¹).
        Par défaut 24.0.
    qmaxinst : float, optional
        Limite supérieure de Q pour le fit du fond polynomial. Par défaut
        égal à ``qmax``.
    rmin, rmax, rstep : float, optional
        Étendue et pas de l'axe r (Å). Par défaut 0.0, 50.0, 0.01.
    rpoly : float, optional
        Paramètre de contrôle du degré polynomial (convention PDFgetX3).
        Par défaut 1.4.
    Lorch : bool, optional
        Appliquer la fonction de modification de Lorch. Par défaut ``True``.
    outputfile : str, optional
        Chemin du fichier .gr de sortie. Si ``None``, aucun fichier n'est
        écrit.
    plot : bool, optional
        Afficher les graphes de diagnostic. Par défaut ``False``.

    Returns
    -------
    r : ndarray
        Axe de distance réelle en Å.
    G : ndarray
        Fonction de distribution de paires réduite G(r) en Å⁻².
    """
    image = np.asarray(image, dtype=float)

    # Intégration azimutale de l'échantillon
    q, I = _integrate_image(image, poni_file, mask=mask, npt=npt)

    # Intégration azimutale de la référence (fond) si fournie
    I_ref = None
    if ref is not None:
        ref_image = _load_ref_image(ref)
        _, I_ref = _integrate_image(ref_image, poni_file, mask=mask, npt=npt)

    # Calcul de la PDF
    r, G = compute_xPDF(
        q, I,
        composition,
        Iref=I_ref,
        bgscale=bgscale,
        qmin=qmin,
        qmax=qmax,
        qmaxinst=qmaxinst,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        rpoly=rpoly,
        Lorch=Lorch,
        plot=plot,
    )

    if outputfile is not None:
        _save_gr(r, G, outputfile, None, composition,
                 bgscale, qmin, qmax, qmaxinst, rmin, rmax, rstep, rpoly)

    return r, G

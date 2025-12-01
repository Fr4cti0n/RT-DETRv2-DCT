#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Ce module contient la classe FrameDCT, qui permet de 
représenter une frame au format DCT (2D) et ses attributs.
"""
# importation des librairies


import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import idct, dct
from math import ceil
from .frame import Frame

class FrameDCT(Frame):

    """
    Classe FrameDCT qui permet de représenter une frame 
    de la vidéo en utilisant la transformée en cosinus discrète 2D (DCT).
    La classe FrameDCT hérite de la classe Frame.

    La classe FrameDCT possède les attributs suivants:
        - frame: la frame
        - type_frame: le type de la frame
        - taille: la taille de la frame
        - vecteur_mouvement : matrice des vecteurs de mouvement de la frame

    La classe FrameDCT possède les méthodes suivantes:
        - idct_2d_bloc: permet de calculer la DCT inverse d'un bloc 8x8
        - idct_2d: permet de calculer la DCT inverse de la frame
        - affichage_vecteur_mouvement : permet d'afficher les vecteurs de mouvement de la frame
        - affichage_dct: permet de afficher la frame
    """

    def __init__(self, frame: np.ndarray, type_frame: str,
                 taille: tuple, vecteur_mouvement=None,
                 cb_dct: np.ndarray | None = None,
                 cr_dct: np.ndarray | None = None) -> None:
        """
        Constructeur de la classe FrameDCT qui permet d'initialiser les attributs de la classe.
Les vecteur_mouvement sont mis sous forme de matrice avant d'être placé dans l'attribut.

        Parametres
        ----------
        - self: FrameDCT
        - frame: np.ndarray
            la frame
        - type_frame: str
            le type de la frame
        - taille: tuple
            la taille de la frame

        Returns
        -------
        - None
        """

        super().__init__(frame, type_frame, taille)
        # vecteur de mouvement est une matrice de taille (n//16,p//16,3)
        # contenant les vecteurs de mouvement (dx, dy, target)
        # target indique de quellle frame vient le mouvement : la suivante ou la précédente
        # dans notre cas target = -1 dans tous les cas on a que des p-frame
        if vecteur_mouvement is None:
            self.vecteur_mouvement = None
        else:  # On met les vecteurs de mouvement sous forme de matrice
            mv_frame, mv_target = vecteur_mouvement
            h, w, _ = mv_frame.shape

            temp_frame = np.zeros((h * 16, w * 16, 3))
            for h_mv in range(h):
                for w_mv in range(w):
                    temp_frame[h_mv*16:(h_mv+1)*16, w_mv *
                               16:(w_mv+1)*16, :2] = mv_frame[h_mv, w_mv]
                    temp_frame[h_mv*16:(h_mv+1)*16, w_mv *
                               16:(w_mv+1)*16, 2] = mv_target[h_mv, w_mv]
            self.vecteur_mouvement = temp_frame
        self.cb_dct = cb_dct
        self.cr_dct = cr_dct
        self._chroma_target_shape = (ceil(taille[0] / 2), ceil(taille[1] / 2))

    def affichage_vecteur_mouvement(self):
        """ Affiche les vecteurs de mouvement

        Parameters
        ----------
        self : FrameDCT

        Returns
        -------
        None
        """
        if self.vecteur_mouvement is not None:
            rgb_h, rgb_w = self.taille
            temp_frame = self.vecteur_mouvement

            # plt.title("Vecteurs de mouvement")
            plt.plot(0, 0, "None")
            plt.plot(rgb_w, rgb_h, "None")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('auto')
            for h_mv in range(0, rgb_h, 16):
                for w_mv in range(0, rgb_w, 16):
                    if any(v != 0 for v in temp_frame[h_mv, w_mv, :2]):
                        dw, dh, target = temp_frame[h_mv, w_mv]
                        destination_h = h_mv + 8  # La destination du MV est le centre du macro-bloc
                        destination_w = w_mv + 8
                        c = 'g'if target == -1 else 'r'
                        plt.quiver(destination_w-dw, destination_h-dh, dw,
                                   dh, angles='xy', scale_units='xy', scale=1, color=c)

    def affichage_dct(self) -> None:
        """
        Méthode qui permet de afficher la frame.

        Parametres
        ----------
        - self: FrameDCT

        Returns
        -------
        - None
        """

        plt.imshow(self.frame, 'gray', vmin=0, vmax=1880)

    def idct_2d_bloc(self, bloc: np.ndarray) -> np.ndarray:
        """
        Méthode qui permet de calculer la DCT inverse d'un bloc 8x8.

        Parametres
        ----------
        - self: FrameDCT
        - bloc: np.ndarray
            un bloc 8x8 de la composante Y de la frame

        Returns
        -------
        - idct_2d: np.ndarray
            la DCT inverse du bloc 8x8
        """

        idct_axe_0 = idct(bloc, type=2, n=None, axis=0,
                          norm='ortho', overwrite_x=False)
        idct_2D = idct(idct(bloc.T,type=2, norm = 'ortho').T, norm = 'ortho')
        return idct_2D
    
    @staticmethod
    def _prepare_macroblocks(macroblocks: np.ndarray) -> np.ndarray:
        """Retourne une carte 2D des macroblocs à partir des données extraites."""
        if isinstance(macroblocks, np.ndarray) and macroblocks.ndim == 3 and macroblocks.shape[-1] == 1:
            return macroblocks[:, :, 0]
        return macroblocks

    @staticmethod
    def _extract_previous_planes(previous_frame) -> dict:
        """Retourne les plans Y/Cb/Cr disponibles pour la frame précédente."""

        if previous_frame is None:
            return {"y": None, "cb": None, "cr": None}

        if isinstance(previous_frame, dict):
            return {
                "y": previous_frame.get("y"),
                "cb": previous_frame.get("cb"),
                "cr": previous_frame.get("cr"),
            }

        if hasattr(previous_frame, "frame"):
            cb_plane = getattr(previous_frame, "cb_padded", None)
            if cb_plane is None:
                cb_plane = getattr(previous_frame, "cb", None)
            cr_plane = getattr(previous_frame, "cr_padded", None)
            if cr_plane is None:
                cr_plane = getattr(previous_frame, "cr", None)
            return {
                "y": previous_frame.frame,
                "cb": cb_plane,
                "cr": cr_plane,
            }

        return {"y": np.asarray(previous_frame), "cb": None, "cr": None}

    def _idct_residual_from_matrix(self,
                                    matrix: np.ndarray,
                                    top: int,
                                    left: int,
                                    height: int,
                                    width: int) -> np.ndarray:
        """Applique l'iDCT bloc par bloc sur une zone rectangulaire."""

        residual = np.zeros((height, width), dtype=np.float32)
        for block_y in range(0, height, 8):
            for block_x in range(0, width, 8):
                sub_h = min(8, height - block_y)
                sub_w = min(8, width - block_x)
                dct_block = matrix[top + block_y:top + block_y + sub_h,
                                   left + block_x:left + block_x + sub_w]
                working_block = np.zeros((8, 8), dtype=np.float64)
                working_block[:sub_h, :sub_w] = dct_block
                spatial_block = self.idct_2d_bloc(working_block)
                residual[block_y:block_y+sub_h, block_x:block_x+sub_w] = spatial_block[:sub_h, :sub_w]
        return residual

    @staticmethod
    def _bilinear_block(prev_plane: np.ndarray,
                        origin_x: float,
                        origin_y: float,
                        width: int,
                        height: int) -> np.ndarray:
        """Echantillonne un bloc avec interpolation bilinéaire (demi-pixel)."""

        if prev_plane is None:
            return np.zeros((height, width), dtype=np.float32)

        h, w = prev_plane.shape
        block = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            src_y = float(np.clip(origin_y + y, 0.0, h - 1.0))
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, h - 1)
            dy = src_y - y0
            for x in range(width):
                src_x = float(np.clip(origin_x + x, 0.0, w - 1.0))
                x0 = int(np.floor(src_x))
                x1 = min(x0 + 1, w - 1)
                dx = src_x - x0

                top_val = (1.0 - dx) * prev_plane[y0, x0] + dx * prev_plane[y0, x1]
                bottom_val = (1.0 - dx) * prev_plane[y1, x0] + dx * prev_plane[y1, x1]
                block[y, x] = (1.0 - dy) * top_val + dy * bottom_val
        return block

    @staticmethod
    def _motion_compensate(prev_plane: np.ndarray | None,
                           dest_x: int,
                           dest_y: int,
                           mv_x: int,
                           mv_y: int,
                           width: int,
                           height: int,
                           subsampling_x: float,
                           subsampling_y: float,
                           hpel_flag: int | None = None) -> np.ndarray:
        """Réalise la compensation de mouvement pour un bloc donné."""

        # mv_x/mv_y fournis par l'extracteur sont exprimés en demi-pixels et
        # représentent le déplacement de la source vers la destination avec un
        # signe opposé à la convention utilisée dans l'implémentation historique.
        # On annule donc le signe ici pour garder la même convention que l'ancien
        # pipeline (delta = destination - source) avant d'appliquer l'interpolation.

        if prev_plane is None:
            return np.zeros((height, width), dtype=np.float32)

        safe_subsampling_x = max(subsampling_x, 1.0)
        safe_subsampling_y = max(subsampling_y, 1.0)

        delta_x = -(mv_x / 2.0) / safe_subsampling_x
        delta_y = -(mv_y / 2.0) / safe_subsampling_y

        origin_x = dest_x - delta_x
        origin_y = dest_y - delta_y

        # Le drapeau hpel est conservé uniquement pour compatibilité si jamais
        # l'extracteur encode des offsets fractionnaires qui ne sont pas déjà
        # reflétés dans mv_x/mv_y. Avec l'interpolation bilinéaire, aucune
        # correction supplémentaire n'est nécessaire, mais on laisse la variable
        # dans la signature pour éviter de casser l'API.
        _ = hpel_flag

        return FrameDCT._bilinear_block(prev_plane, origin_x, origin_y, width, height)

    def _decode_luma(self,
                     macroblocks: np.ndarray,
                     prev_y: np.ndarray | None,
                     motion_x: np.ndarray | None,
                     motion_y: np.ndarray | None,
                     hpel_map: np.ndarray | None) -> np.ndarray:
        """Reconstruit le plan Y en tenant compte des vecteurs de mouvement."""

        height, width = self.taille
        mb_rows = max(1, (height + 15) // 16)
        mb_cols = max(1, (width + 15) // 16)

        if macroblocks is None:
            macroblocks = np.ones((mb_rows, mb_cols), dtype=np.int32)

        plane = np.zeros((height, width), dtype=np.float32)
        for mb_y in range(mb_rows):
            y0 = mb_y * 16
            block_h = min(16, height - y0)
            for mb_x in range(mb_cols):
                x0 = mb_x * 16
                block_w = min(16, width - x0)

                residual = self._idct_residual_from_matrix(self.frame, y0, x0, block_h, block_w)

                mb_flag = macroblocks[min(mb_y, macroblocks.shape[0]-1),
                                      min(mb_x, macroblocks.shape[1]-1)]
                if self.type_frame == 'k' or mb_flag == 1 or prev_y is None:
                    block = residual
                else:
                    mvx = 0 if motion_x is None else motion_x[min(mb_y, motion_x.shape[0]-1),
                                                              min(mb_x, motion_x.shape[1]-1)]
                    mvy = 0 if motion_y is None else motion_y[min(mb_y, motion_y.shape[0]-1),
                                                              min(mb_x, motion_y.shape[1]-1)]
                    hpel_flag = 0 if hpel_map is None else hpel_map[min(mb_y, hpel_map.shape[0]-1),
                                                                     min(mb_x, hpel_map.shape[1]-1)]
                    prediction = self._motion_compensate(prev_y, x0, y0, mvx, mvy,
                                                         block_w, block_h,
                                                         subsampling_x=1,
                                                         subsampling_y=1,
                                                         hpel_flag=hpel_flag)
                    block = prediction + residual

                plane[y0:y0+block_h, x0:x0+block_w] = block

        return np.clip(np.round(plane), 0, 255).astype(np.float32)

    def _decode_chroma(self,
                       chroma_dct: np.ndarray | None,
                       macroblocks: np.ndarray,
                       prev_chroma: np.ndarray | None,
                       motion_x: np.ndarray | None,
                       motion_y: np.ndarray | None,
                       hpel_map: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reconstruit un plan chroma (Cb ou Cr) ainsi que sa version recadrée."""

        if chroma_dct is None:
            return None, None

        height, width = chroma_dct.shape
        mb_rows = max(1, (height + 7) // 8)
        mb_cols = max(1, (width + 7) // 8)

        plane_full = np.zeros((height, width), dtype=np.float32)
        subsampling_y = max(1.0, self.taille[0] / float(height)) if height else 2.0
        subsampling_x = max(1.0, self.taille[1] / float(width)) if width else 2.0
        for mb_y in range(mb_rows):
            y0 = mb_y * 8
            block_h = min(8, height - y0)
            for mb_x in range(mb_cols):
                x0 = mb_x * 8
                block_w = min(8, width - x0)

                residual = self._idct_residual_from_matrix(chroma_dct, y0, x0, block_h, block_w)

                mb_flag = macroblocks[min(mb_y, macroblocks.shape[0]-1),
                                      min(mb_x, macroblocks.shape[1]-1)]
                if self.type_frame == 'k' or mb_flag == 1 or prev_chroma is None:
                    block = residual
                else:
                    mvx = 0 if motion_x is None else motion_x[min(mb_y, motion_x.shape[0]-1),
                                                              min(mb_x, motion_x.shape[1]-1)]
                    mvy = 0 if motion_y is None else motion_y[min(mb_y, motion_y.shape[0]-1),
                                                              min(mb_x, motion_y.shape[1]-1)]
                    hpel_flag = 0 if hpel_map is None else hpel_map[min(mb_y, hpel_map.shape[0]-1),
                                                                     min(mb_x, hpel_map.shape[1]-1)]
                    prediction = self._motion_compensate(prev_chroma, x0, y0, mvx, mvy,
                                                         block_w, block_h,
                                                         subsampling_x=subsampling_x,
                                                         subsampling_y=subsampling_y,
                                                         hpel_flag=hpel_flag)
                    block = prediction + residual

                plane_full[y0:y0+block_h, x0:x0+block_w] = block

        plane_cropped = self._crop_chroma_plane(plane_full)
        if plane_cropped is not None:
            plane_cropped = np.clip(np.round(plane_cropped), 0, 255).astype(np.float32)
        plane_full = np.clip(np.round(plane_full), 0, 255).astype(np.float32)
        return plane_full, plane_cropped

    def _crop_chroma_plane(self, plane: np.ndarray | None) -> np.ndarray | None:
        if plane is None:
            return None
        target_h, target_w = self._chroma_target_shape
        return plane[:target_h, :target_w]

    @staticmethod
    def _forward_dct_plane(plane: np.ndarray | None) -> np.ndarray | None:
        """Calcule la DCT bloc par bloc d'un plan spatial."""

        if plane is None:
            return None
        height, width = plane.shape
        if height % 8 != 0 or width % 8 != 0:
            pad_h = (8 - height % 8) % 8
            pad_w = (8 - width % 8) % 8
            plane = np.pad(plane, ((0, pad_h), (0, pad_w)), mode='edge')
            height, width = plane.shape

        blocks_y = height // 8
        blocks_x = width // 8

        reshaped = plane.reshape(blocks_y, 8, blocks_x, 8).transpose(0, 2, 1, 3)
        blocks = reshaped.reshape(-1, 8, 8)

        blocks = dct(blocks, type=2, norm='ortho', axis=1)
        blocks = dct(blocks, type=2, norm='ortho', axis=2)

        dct_frame = blocks.reshape(blocks_y, blocks_x, 8, 8).transpose(0, 2, 1, 3)
        return dct_frame.reshape(height, width)

    @staticmethod
    def build_dct_state(component) -> dict:
        """Construit l'état DCT (Y/Cb/Cr) à partir d'une frame reconstruite."""

        state = {"y": component.dct_2d().frame, "cb": None, "cr": None}
        cb_source = getattr(component, "cb_padded", None)
        if cb_source is None:
            cb_source = getattr(component, "cb", None)
        cr_source = getattr(component, "cr_padded", None)
        if cr_source is None:
            cr_source = getattr(component, "cr", None)
        state["cb"] = FrameDCT._forward_dct_plane(cb_source)
        state["cr"] = FrameDCT._forward_dct_plane(cr_source)
        return state
    def idct_2d(self,
                macroblocks,
                previous_frame,
                motion_x: np.ndarray | None = None,
                motion_y: np.ndarray | None = None,
                hpel: np.ndarray | None = None) -> np.ndarray:
        """Calcule l'iDCT de la frame Y ainsi que des plans chroma si disponibles."""

        # pylint: disable=import-outside-toplevel
        # cet import est une solution pour éviter un probleme d'importation circulaire
        from compression.frame.composante_y import ComposanteY

        macroblocks_map = self._prepare_macroblocks(macroblocks)
        if macroblocks_map is None:
            height, width = self.taille
            macroblocks_map = np.ones(((height + 15) // 16, (width + 15) // 16), dtype=np.int32)

        prev_planes = self._extract_previous_planes(previous_frame)

        motion_x_map = None if motion_x is None else np.asarray(motion_x)
        motion_y_map = None if motion_y is None else np.asarray(motion_y)

        y_plane = self._decode_luma(macroblocks_map,
                        prev_planes["y"],
                        motion_x_map,
                        motion_y_map,
                        hpel)

        cb_full, cb_plane = self._decode_chroma(self.cb_dct, macroblocks_map,
                            prev_planes["cb"],
                            motion_x_map,
                            motion_y_map,
                            hpel)
        cr_full, cr_plane = self._decode_chroma(self.cr_dct, macroblocks_map,
                            prev_planes["cr"],
                            motion_x_map,
                            motion_y_map,
                            hpel)

        composante = ComposanteY(y_plane, self.type_frame, (self.taille[0], self.taille[1]),
                                 cb=cb_plane, cr=cr_plane)
        composante.cb_padded = cb_full
        composante.cr_padded = cr_full
        return composante
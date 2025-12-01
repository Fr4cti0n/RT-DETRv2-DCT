#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce module contient la classe Frame, qui permet de représenter une frame et ses attributs.
"""
# Importation des librairies
import numpy as np


class Frame:
    """
    Classe Frame qui permet de représenter une frame de la vidéo.

    La classe Frame possède les attributs suivants:
        - frame: la frame
        - type_frame: le type de la frame
        - taille: la taille de la frame

    La classe Frame possède les méthodes suivantes:
        - __add__: permet d'additionner deux frames 
        - __sub__: permet de soustraire deux frames
        - __getitem__: permet de récupérer un élément de la frame
        - __str__: permet de représenter la frame
    """

    def __init__(self, frame: np.ndarray, type_frame: str, taille: tuple) -> None:
        """
        Constructeur de la classe Frame qui permet d'initialiser les attributs de la classe.

        Parametres
        ----------
        - self: Frame
        - frame: np.ndarray
            la frame
        - type_frame: str
            le type de la frame
        - taille: tuple
            la taille de la frame

        Returns
        --------
        - None
        """

        self._frame = frame
        self._type_frame = type_frame
        self._taille = taille

    @property
    def frame(self) -> np.ndarray:
        """
        Méthode qui permet de récupérer la frame.

        Parametres
        -----------
        - self: Frame

        Returns
        --------
        - self._frame: np.ndarray
        """
        return self._frame

    @property
    def type_frame(self) -> str:
        """
        Méthode qui permet de récupérer le type de la frame.

        Parametres
        -----------
        - self: Frame

        Returns
        --------
         self._type_frame: str
        """
        return self._type_frame

    @property
    def taille(self) -> tuple:
        """
        Méthode qui permet de récupérer la taille de la frame.

        Parametres
        -----------
        - self: Frame

        Returns:
        --------
        - self._taille: tuple
        """

        return self._taille

    @frame.setter
    def frame(self, new_frame):
        self._frame = new_frame

    def __add__(self, frame: np.ndarray) -> np.ndarray:
        """
        Méthode qui permet d'additionner deux frames.

        Parametres
        -----------
        - self: Frame
        - frame: np.ndarray
            la frame à additionner

        Returns
        --------
        - self.__class__: np.ndarray
                la frame additionnée
        """

        return self.__class__(self.frame + frame.frame, self.type_frame, self.taille)

    def __sub__(self, frame: np.ndarray) -> np.ndarray:
        """
        Méthode qui permet de soustraire deux frames.

        Parametres
        ----------
        - self: Frame
        - frame: np.ndarray
            la frame à soustraire

        Returns
        --------
        - self.__class__: np.ndarray
                la frame soustraite
        """
        return self.__class__(self.frame - frame.frame, self.type_frame, self.taille)

    def __getitem__(self, cle: np.ndarray) -> np.ndarray:
        """
        Méthode qui permet de récupérer un élément de la frame.

        Parametres
        ----------
        - self: Frame
        - cle: np.ndarray
            la clé de la frame

        Returns
        --------
        - self.frame[cle]: np.ndarray
                l'élément de la frame
        """

        return self.frame[cle]

    def __eq__(self, other):
        """
        Méthode qui permet de vérifier l'égalité entre deux frames.

        Paramètres
        ----------
        - self: Frame
        - other: Frame
            L'autre frame à comparer.

        Returns
        --------
        - bool
            True si les frames sont égales, False sinon.
        """
        if isinstance(other, type(self)):
            return (np.allclose(self.frame, other.frame, rtol=1880*0.01) and
                    self.type_frame == other.type_frame and
                    self.taille == other.taille)
        return False

    def __str__(self) -> str:
        """
        Méthode qui permet de représenter la frame.

        Parametres
        ----------
        - self: Frame

        Returns
        --------
        - chaine: str
                la représentation de la frame
        """
        chaine = ''
        hauteur, largeur = self.taille
        for i in range(hauteur//8):
            for j in range(largeur//8):
                chaine = chaine + '\n' + \
                    str(self.frame[i*8:(i+1)*8, j*8:(j+1)*8])
        return chaine

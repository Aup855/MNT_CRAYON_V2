# Solveur thermique 1D radial : crayon combustible nucléaire : v2 (convection en surface)

**Ajout de la condition de Robin (échange convectif) sur la surface externe de la pastille, en remplacement de la température de surface imposée de la v1.**

> La v1 imposait directement la température de surface (`T_surface`) pour isoler et valider le cœur du schéma numérique. Cette hypothèse n'est pas physique : dans un crayon réel, ce qui est connu côté fluide caloporteur, c'est sa température et un coefficient d'échange et non la température de la surface elle-même, qui est une **conséquence** du calcul et non une donnée d'entrée. La v2 lève cette hypothèse : la surface externe devient une inconnue de plus du système, pilotée par un bilan convectif (condition de Robin).

---

## Sommaire

1. [Présentation du problème](#1-présentation-du-problème)
2. [Hypothèses simplificatrices](#2-hypothèses-simplificatrices)
3. [Méthodologie du code](#3-méthodologie-du-code)
4. [Résultats](#4-résultats)
5. [Points critiques](#5-points-critiques)
6. [Tests](#6-tests)
7. [Conclusion](#7-conclusion)

---

## 1. Présentation du problème

Le point le plus chaud du crayon reste sur l'axe (`r = 0`), comme en v1 : la chaleur générée par fission traverse la pastille par conduction radiale. Ce qui change en v2, c'est ce qui se passe *à* la surface (`r = R`) : au lieu d'y imposer une température, on y impose la **continuité du flux** entre conduction (côté solide) et convection (côté fluide caloporteur).

<p align="center">
  <img src="docs/img/schema_probleme_v2.png" width="900" alt="Schéma du problème v2 : coupe radiale avec convection en surface, et profil de température attendu (parabole décalée)">
</p>

*Gauche : coupe radiale, la surface n'est plus à température fixée mais échange par convection avec le fluide (`h`, `T_fluide`). Droite : le profil reste une parabole (conduction interne inchangée), mais toute la courbe est translatée du saut convectif `q'''R/(2h)`.*

L'équation résolue à l'intérieur de la pastille est inchangée par rapport à la v1 :

$$\rho c_p \frac{\partial T}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(r\,k\,\frac{\partial T}{\partial r}\right) + \dot q'''$$

Seule la condition limite en surface change :
- **au centre (`r = 0`)** : symétrie (inchangé : émerge de la géométrie ) ;
- **en surface (`r = R`)** : condition de **Robin** : $-k\,\dfrac{\partial T}{\partial r}\Big|_R = h\,(T(R) - T_{fluide})$, au lieu d'un Dirichlet imposé.

---

## 2. Hypothèses simplificatrices

La v2 lève une hypothèse de la v1 (la surface imposée) et en garde volontairement plusieurs autres, pour continuer à isoler un seul changement à la fois :

| # | Hypothèse | Portée |
|---|-----------|--------|
| 1 | **Axisymétrie + invariance axiale** | Reprise à l'identique de la v1 : problème 1D radial pur. |
| 2 | **Monomatériau** | Toujours pastille pleine homogène : pas de jeu ni de gaine. |
| 3 | **Propriétés constantes** | `k`, `ρc_p` indépendants de `T`. |
| 4 | **Coefficient d'échange `h` constant et uniforme** | Pas de corrélation d'ébullition nucléée ni de dépendance à l'écoulement. |
| 5 | **Source volumique uniforme (cas de référence)** | Comme en v1, avec profil `source(r)` arbitraire accepté par le solveur. |
| 6 | **Pas d'évolution géométrique** | Géométrie figée dans le temps (pas de gonflement, fissuration, relocalisation). |

---


## 3. Méthodologie du code

**Reprise intégrale du solveur v1** : volumes finis vertex-centered, schéma de Crank-Nicolson, matrice tridiagonale assemblée une seule fois puis résolue en `O(N)` via `scipy.linalg.solve_banded`. 
Le nœud central, les nœuds intérieurs et toute la géométrie (`V_i`, `A_i`) sont repris à l'identique. Le détail de cette base est expliqué dans le [README de la v1](https://github.com/Aup855/MNT_CRAYON_V2/blob/v1/README.md).

**Ce qui change : une seule ligne, la dernière.** Tout le reste du système (nœud central, nœuds intérieurs) est identique à la v1. 
Seules la dernière ligne de la matrice et la dernière ligne du second membre sont modifiées, pour faire apparaître la convection en surface.

L'idée clé est simple : un flux convectif s'écrit exactement comme un flux conductif, c'est à dire une **conductance multipliée par un écart de température**. Il suffit de comparer les deux termes :

```
flux conductif interne (entre deux nœuds du maillage) :
    a_W * (T_N-2 - T_N-1)          avec  a_W = A_(N-3/2) * k / dr

flux convectif externe (entre le dernier nœud et le fluide) :
    h*A_R * (T_fluide - T_N-1)     avec  A_R = 2*pi*R (surface externe réelle)
```

La forme mathématique est identique des deux côtés : une conductance (en W/K) fois un écart de température. 
Seule change la nature de ce à quoi on est relié : la conductance interne relie deux nœuds du maillage entre eux, alors que la conductance convective relie le dernier nœud à un **réservoir extérieur** de température fixe, `T_fluide`.

C'est cette analogie qui permet de traiter Robin sans complexité supplémentaire : il ne s'agit pas d'un cas particulier à coder à part, mais d'une conductance de plus dans le même bilan. 
Concrètement, `h*A_R` s'ajoute au terme diagonal de la ligne `N-1` (elle « freine » l'évolution de `T_N-1`, exactement comme le ferait une conductance interne), et `h*A_R*T_fluide` s'ajoute au second membre comme un forçage constant.

**Point de vigilance.** Dans le second membre de Crank-Nicolson, chaque terme est normalement pondéré par un facteur 1/2 (moyenne entre l'instant `n` et l'instant `n+1`). Ce n'est *pas* le cas pour `h*A_R*T_fluide` : comme `T_fluide` est une donnée constante du problème, identique à `n` et à `n+1`, sa contribution s'écrit `1/2*h*A_R*T_fluide + 1/2*h*A_R*T_fluide`, et les deux moitiés se recombinent naturellement en un terme plein.

**API** : `PastilleV2(alpha, k_th, R, N, dt, h_conv, T_fluide, T_init, source)`, puis `.step()` / `.solve(n_steps)` ; diagnostic de bilan de puissance via `flux_sortant_frontiere()`, désormais calculé sur le disque complet (voir §5 pour la nuance avec la v1)

---

## 4. Résultats

Cas de validation, mêmes valeurs REP que la v1 (`R = 4,1 mm`, `k = 3 W/(m·K)`, `q''' = 3,8×10⁸ W/m³`), avec `h = 3,5×10⁴ W/(m²·K)` et `T_fluide = 310 °C`. En régime permanent, la solution analytique est une parabole **décalée** du saut convectif :

$$T(r) = \underbrace{T_{fluide} + \frac{\dot q'''R}{2h}}_{\text{saut convectif}} + \underbrace{\frac{\dot q'''}{4k}\left(R^2 - r^2\right)}_{\text{parabole (identique à v1)}}$$

<p align="center">
  <img src="docs/img/validation_v2.png" width="620" alt="Comparaison solution numérique vs parabole décalée analytique">
</p>

| Grandeur | Valeur numérique | Valeur analytique |
|---|---|---|
| Température de surface | 332,2571 °C | 332,2571 °C |
| Température au centre | 864,5738 °C | — |
| Puissance générée (disque complet) | 20 067,87 W/m | 20 067,87 W/m |
| Puissance sortante (convection) | 20 067,87 W/m | (identique, conservation) |
| Erreur max sur tout le profil | 2,6 × 10⁻¹² °C | — |
| Écart relatif du bilan de puissance | 9,97 × 10⁻¹⁵ | — |

Comme en v1, l'accord à la précision machine confirme que le schéma discret représente exactement une solution polynomiale de degré ≤ 2. Le bilan de puissance est conservé à l'arrondi machine près.

**Test de cohérence entre versions.** Un test supplémentaire, propre à la v2, vérifie que quand `h → ∞`, la condition de Robin dégénère bien en Dirichlet : `PastilleV2` avec `h = 10⁹` et `T_fluide = T_ref` doit redonner *exactement* `PastilleV1` avec `T_surface = T_ref`. Écart maximal mesuré entre les deux implémentations indépendantes : < 0,05 °C — confirmant que Robin est cohérent avec Dirichlet dans son cas limite, sans régression sur le code v1.

---

## 5. Points critiques

Ce que cette v2 **ne** capture **pas**, en plus des limites déjà présentes en v1 (conductivité constante, monomatériau, pas de couplage 2D/3D, pas d'irradiation) :

- **`h` constant et donné, pas calculé.** Le coefficient d'échange convectif dépend en réalité du régime d'écoulement, de la sous-saturation locale, et peut varier fortement en cas d'ébullition nucléée, ici c'est une donnée d'entrée figée, pas le résultat d'une corrélation thermohydraulique.
- **`T_fluide` uniforme et constante.** Aucun couplage avec un bilan enthalpique axial du fluide (le caloporteur s'échauffe le long du crayon) : `T_fluide` est traitée comme un réservoir infini, identique en tout point et à tout instant.
- **Toujours pas de jeu ni de gaine.** La convection remplace directement la conduction dans la pastille.
- **Précision non quantifiée en régime transitoire**, comme en v1 : seules la stabilité et la justesse à l'équilibre sont couvertes par les tests.

---

## 6. Tests

Suite de tests (`pytest test_pastille_v2.py -v`), reprenant les tests hérités de la v1 (stabilité, isotherme sans source, centre maximum, pas de singularité) et ajoutant les tests spécifiques à Robin :

| Test | Ce qu'il protège |
|---|---|
| `test_surface_plus_chaude_que_fluide` | Bon sens physique : le flux ne peut aller que du chaud vers le froid : attrape une erreur de signe sur le terme convectif. |
| `test_parabole_decalee_regime_permanent` | Validation quantitative complète : forme de la parabole *et* position du saut convectif, à mieux que 0,5 °C. |
| `test_bilan_energie_regime_permanent` | Conservation sur le disque **complet**. |
| `test_limite_h_infini_redonne_dirichlet` | Cohérence entre versions : Robin doit dégénérer en Dirichlet quand `h → ∞`, pas dériver vers autre chose. |
| `test_source_nulle_equivaut_none` | Rétrocompatibilité : `source=None` ≡ `source=0`. |
| `test_stabilite_plusieurs_maillages` *(×4)* | Robustesse sur 4 résolutions (N = 21 à 201). |

---

## 7. Conclusion

Cette v2 démontre que le découplage de l'architecture en volumes finis, mis en place dès la v1, tient sa promesse : changer la physique de la condition limite en surface (Dirichlet → Robin) n'a nécessité de modifier **qu'une seule ligne** de la matrice et du second membre, sans toucher au nœud central ni aux nœuds intérieurs. Le test de cohérence `h → ∞` confirme que cette nouvelle implémentation converge vers la v1 dans son cas limite, là où les deux doivent mathématiquement coïncider. 

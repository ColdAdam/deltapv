from deltapv import scales, physics, objects, spline, util
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.patches import Rectangle
import os

PVDesign = objects.PVDesign
Potentials = objects.Potentials
Array = util.Array

COLORS = ["darkorange", "yellow", "limegreen", "cyan", "indigo"]

font_dir = os.path.join(os.path.dirname(__file__), "fonts")
font_files = fm.findSystemFonts(fontpaths=font_dir)
for font_file in font_files:
    fm.fontManager.addfont(font_file)
rcParams["font.family"] = "CMU Serif"
rcParams["font.size"] = 15
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "CMU Serif"
rcParams["mathtext.it"] = "CMU Serif:italic"
rcParams["mathtext.bf"] = "CMU Serif:bold"
rcParams["lines.linewidth"] = 2
rcParams["patch.linewidth"] = 2
rcParams["hatch.linewidth"] = 2
rcParams["figure.figsize"] = (8.0, 5.0)
rcParams["figure.dpi"] = 80
rcParams["savefig.dpi"] = 300


def plot_bars(design: PVDesign=None,
              gui: plt.Figure=None,
              filename=None) -> None:

    plt_bars = gui

    ax1 = plt_bars.subplots()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    ax1.set_xlabel("position / μm")
    ax1.set_ylabel("energy / eV")
    ax1.legend()

    if not design and gui:
        return plt_bars
    else:
        Ec = scales.energy * physics.Ec(design)
        Ev = scales.energy * physics.Ev(design)
        EF = scales.energy * physics.EF(design)
        dim_grid = scales.length * design.grid * 1e4
        if design.PhiM0 > 0 and design.PhiML > 0:
            ax1.margins(x=.2, y=.5)
        else:
            ax1.margins(y=.5)
            ax1.set_xlim(0, dim_grid[-1])

        idx = jnp.concatenate(
            [jnp.array([0]),
             jnp.argwhere(Ec[:-1] != Ec[1:]).flatten() + 1])

        uc = Ec[idx]
        uv = Ev[idx]
        startx = dim_grid[idx]
        starty = uv
        height = uc - uv
        width = jnp.diff(jnp.append(startx, dim_grid[-1]))

        for i in range(startx.size):
            x, y, w, h = startx[i], starty[i], width[i], height[i]
            rect = Rectangle((x, y),
                             w,
                             h,
                             color=COLORS[i % len(COLORS)],
                             linewidth=0,
                             alpha=.2)
            ax1.add_patch(rect)
            ax1.text(x + w / 2,
                     y + h + .1,
                     round(y + h, 2),
                     ha="center",
                     va="bottom")
            ax1.text(x + w / 2, y - .1, round(y, 2), ha="center", va="top")

        ax1.plot(dim_grid, EF, linestyle="--", color="black", label="$E_{F}$")

        if design.PhiM0 > 0:
            phim0 = -design.PhiM0 * scales.energy
            xstart, _ = ax1.get_xlim()
            width = -xstart
            height = .2
            ystart = phim0 - height / 2
            rect = Rectangle((xstart, ystart),
                             width,
                             height,
                             color="red",
                             linewidth=0,
                             alpha=.2)
            ax1.add_patch(rect)
            ax1.text(xstart + width / 2,
                     ystart + height + 0.1,
                     round(phim0, 2),
                     ha="center",
                     va="bottom")
            ax1.text(xstart + width / 2,
                     ystart - 0.1,
                     "contact",
                     ha="center",
                     va="top")
            ax1.axhline(y=phim0,
                        xmin=0,
                        xmax=1 / 7,
                        linestyle="--",
                        color="black",
                        linewidth=2)
            ax1.axvline(dim_grid[0], color="white", linewidth=2)
            ax1.axvline(dim_grid[0],
                        color="lightgray",
                        linewidth=2,
                        linestyle="dashed")

        if design.PhiML > 0:
            phiml = -design.PhiML * scales.energy
            xstart = dim_grid[-1]
            _, xend = ax1.get_xlim()
            width = xend - xstart
            height = .2
            ystart = phiml - height / 2
            rect = Rectangle((xstart, ystart),
                             width,
                             height,
                             color="blue",
                             linewidth=0,
                             alpha=.2)
            ax1.add_patch(rect)
            ax1.text(xstart + width / 2,
                     ystart + height + 0.1,
                     round(phiml, 2),
                     ha="center",
                     va="bottom")
            ax1.text(xstart + width / 2,
                     ystart - 0.1,
                     "contact",
                     ha="center",
                     va="top")
            ax1.axhline(y=phiml,
                        xmin=6 / 7,
                        xmax=1,
                        linestyle="--",
                        color="black",
                        linewidth=2)
            ax1.axvline(dim_grid[-1], color="white", linewidth=2)
            ax1.axvline(dim_grid[-1],
                        color="lightgray",
                        linewidth=2,
                        linestyle="dashed")

        posline = jnp.argwhere(design.Ndop[:-1] != design.Ndop[1:]).flatten()

        for idx in posline:
            vpos = (dim_grid[idx] + dim_grid[idx + 1]) / 2
            ax1.axvline(vpos, color="white", linewidth=4)
            ax1.axvline(vpos, color="lightgray",
                        linewidth=2, linestyle="dashed")

        ax1.set_ylim(jnp.min(uv) * 1.5, 0)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if not gui:
            plt.show()


def plot_band_diagram(design: PVDesign=None,
                      pot: Potentials=None,
                      gui: plt.Figure=None,
                      eq=False,
                      filename=None) -> None:

    plt_band = gui
    ax1 = plt_band.add_subplot(111)

    ax1.set_xlabel("position / μm")
    ax1.set_ylabel("energy / eV")

    if not design and not pot and gui:
        return plt_band
    else:
        Ec = -scales.energy * (design.Chi + pot.phi)
        Ev = -scales.energy * (design.Chi + design.Eg + pot.phi)
        x = scales.length * design.grid * 1e4

        ax1.plot(x,
                 Ec,
                 color="lightcoral",
                 label="conduction band",
                 linestyle="dashed")
        ax1.plot(x,
                 Ev,
                 color="cornflowerblue",
                 label="valence band",
                 linestyle="dashed")

        if not eq:
            ax1.plot(x,
                     scales.energy * pot.phi_n,
                     color="lightcoral",
                     label="e- quasi-Fermi energy")
            ax1.plot(x,
                     scales.energy * pot.phi_p,
                     color="cornflowerblue",
                     label="hole quasi-Fermi energy")
        else:
            ax1.plot(x,
                     scales.energy * pot.phi_p,
                     color="lightgray",
                     label="Fermi level")

        ax1.legend()
        ax1.set_xlim(0, x[-1])

        plt_band.tight_layout()
        if filename is not None:
            plt_band.savefig(filename)
        if not gui:
            plt_band.show()


def plot_iv_curve(voltages: Array=None,
                  currents: Array=None,
                  gui: plt.Figure=None,
                  filename=None) -> None:

    plt_iv = gui
    ax1 = plt_iv.add_subplot(111)

    ax1.set_xlabel("bias / V")
    ax1.set_ylabel("current density / mA/cm$^2$")

    if not voltages and not currents and gui:
        return plt_iv
    else:
        currents = 1e3 * currents  # A / cm^2 -> mA / cm^2
        coef = spline.qspline(voltages, currents)

        a, b, c = coef
        al, bl, cl = a[-1], b[-1], c[-1]
        discr = jnp.sqrt(bl ** 2 - 4 * al * cl)
        x0 = (-bl - discr) / (2 * al)
        voc = jnp.clip(x0, voltages[-2], voltages[-1])

        vint = jnp.linspace(0, voc, 500)
        jint = spline.predict(vint, voltages, coef)
        idx = jnp.argmax(vint * jint)
        vmax = vint[idx]
        jmax = jint[idx]
        pmax = vmax * jmax
        p0 = jnp.sum(jnp.diff(vint) * (jint[:-1] + jint[1:]) / 2)
        FF = pmax / p0 * 100  # %

        rect = Rectangle((0, 0),
                         vmax,
                         jmax,
                         fill=False,
                         edgecolor="lightgray",
                         hatch="/",
                         linestyle="--")
        ax1.text(vmax / 2,
                 jmax / 2,
                 f"$FF = {round(FF, 2)}\%$\n$MPP = {round(pmax * 10, 2)}$ W/m$^2$",  # noqa
                 ha="center",
                 va="center")
        ax1.gca().add_patch(rect)
        ax1.plot(vint, jint, color="black")
        ax1.scatter(voltages, currents, color="black", marker=".")

        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        plt_iv.tight_layout()
        if filename is not None:
            plt_iv.savefig(filename)
        if not gui:
            plt_iv.show()


def plot_charge(design: PVDesign=None,
                pot: Potentials=None,
                gui: plt.Figure=None,
                filename=None):

    plt_charge = gui
    ax1 = plt_charge.add_subplot(111)

    ax1.set_yscale("log")
    ax1.set_xlabel("position / μm")
    ax1.set_ylabel("density / cm$^{-3}$")

    if not design and not pot and gui:
        return plt_charge
    else:
        n = scales.density * physics.n(design, pot)
        p = scales.density * physics.p(design, pot)

        x = scales.length * design.grid * 1e4

        ax1.plot(x, n, label="electron", color="lightcoral")
        ax1.plot(x, p, label="hole", color="cornflowerblue")

        ax1.legend()
        ax1.set_xlim(0, x[-1])

        plt_charge.tight_layout()
        if filename is not None:
            plt_charge.savefig(filename)
        if not gui:
            plt_charge.show()

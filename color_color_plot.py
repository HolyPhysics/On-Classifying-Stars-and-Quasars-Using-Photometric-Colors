from preprocessing import processed_data
# from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt


X, y, star_data, quasar_data = processed_data()
# print(star_data.columns)
# print(quasar_data.columns)

star_gr = star_data["g_r_magnitude"]
star_ug = star_data["u_g_magnitude"]

quasar_gr = quasar_data["g_r_magnitude"]
quasar_ug = quasar_data["u_g_magnitude"]


figure, ax_main = plt.subplots(figsize=(9.5, 7.5))
ax_main.scatter(star_ug, star_gr, color="green", label="Stars")
ax_main.scatter(quasar_ug, quasar_gr, color="black", label="Quasars")

ax_main.set_xlabel(r"$u-g$ magnitude")
ax_main.set_ylabel(r"$g-r$ magnitude")
ax_main.set_title(r" $g-r$ against $u-g$. ")
ax_main.set_ylim(-0.5, 1)
ax_main.set_xlim(-0.5, 2)
ax_main.legend(loc="best")
figure.tight_layout()



if __name__ == "__main__":
# figure.savefig("/content/drive/MyDrive/Colab Images/quasar_star_color_diagram.png") # I'll have to put this in my slide.
    plt.show()
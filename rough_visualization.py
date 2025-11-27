from main_classification import clean_importation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

color_container_for_analysis, class_integer_container_for_analysis = clean_importation()

g_r_color, r_i_color = color_container_for_analysis[:, 1], color_container_for_analysis[:, 2]

data_to_visualize = pd.DataFrame({"g_r_color": g_r_color, "r_i_color": r_i_color})

# lazy visualization
# sns.pairplot(data_to_visualize)
# plt.show()

figure, ax_main = plt.subplots( figsize =(8.5, 7.5))

ax_main.scatter(r_i_color, g_r_color, linestyle="--", color="black")
ax_main.set_title("$g - r$ against $r - i$ color")
ax_main.set_xlabel("$r - i$ color")
ax_main.set_ylabel("$g - r$ color")
# ax_main.legend(loc="best")
figure.tight_layout()
plt.show()


# Next is to classify them and then color/label the points on the scatter plot using the prediction values !!!


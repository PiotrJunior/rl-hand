from env import make_env
import cv2

# Utwórz zsynchronizowane środowisko
env = make_env(n_envs=1)
obs, info = env.reset()

print("Inicjalizacja udana!")
print("Klucze w obserwacji:", obs.keys())
print("Kształt stanu robota (agent_pos):", obs["agent_pos"].shape)
print("Kształt obrazu z kamery top:", obs["pixels"]["top"].shape)

# Wykonaj jeden losowy krok
losowa_akcja = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(losowa_akcja)
print("Krok wykonany poprawnie. Nagroda:", reward)
# Wyświetl obraz z kamery top
rgb_image = obs["pixels"]["top"][0]
cv2.imwrite("test_top_view.png", rgb_image)  # Zapisz obraz do pliku
# cv2.waitKey(0)  # Czeka na naciśnięcie klawisza

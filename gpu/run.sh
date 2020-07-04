nvcc -o axpy axpy.cu
./axpy

# check gpu
lspci | grep -i nvidia
nvidia-smi

# uninstall the NVIDIA drivers
sudo apt remove --purge '^nvidia-.*'
apt autoremove
apt autoclean
reboot

# Disable/blacklist Nouveau nvidia driver
echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf
echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf

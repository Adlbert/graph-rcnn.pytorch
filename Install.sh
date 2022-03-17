#Python: 3.6.13
#Pytorch: 1.0.1.post2
#CUDA Version: 9.0.176
#torchvision: 0.2.2
#gcc:  5.3.1 
#g++: 5.3.1 

###################################################################
                            #On Restart
sudo mkdir /mnt/ImageFactData
sudo sshfs -o allow_other adhofer@ImageFactData.cs.univie.ac.at:/./dataset /mnt/ImageFactData
conda activate sg
cd graph-rcnn.pytorch/
az group list
az login
###################################################################

cd ~

#disable unattended upgrade
edit /etc/apt/apt.conf.d/20auto-upgrades

    APT::Periodic::Update-Package-Lists "1";
    APT::Periodic::Download-Upgradeable-Packages "1";
    APT::Periodic::AutocleanInterval "7";
    APT::Periodic::Unattended-Upgrade "1";
#set APT::Periodic::Unattended-Upgrade to "0"


sudo adduser admin92
usermod -aG sudo admin92

sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt update
sudo apt-get install nvidia-driver-390
sudo nano /etc/apt/sources.list
#Add following files
#deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
#deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
sudo apt update
sudo apt install g++-5
sudo apt install gcc-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 5
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 5
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
sudo apt-get install libglib2.0-0
sudo apt-get install libsm6
sudo apt-get install git-all
sudo apt-get install sshfs
sudo su
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list

exit
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17


curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash



#wget https://vpn.univie.ac.at/public/share/BIGIPLinuxClient.tgz
#chmod +x BIGIPLinuxClient.tgz
tar -xvf BIGIPLinuxClient.tgz
cd BIGIPLinuxClient
sudo dpkg -i linux_f5cli.x86_64.deb
#f5fpc -s -t vpn.univie.ac.at -u a1209967 -d /etc/ssl/certs/
echo 'f5fpc -s -t vpn.univie.ac.at -u a1209967 -p <vpn_pwd> -d /etc/ssl/certs/'  >> ~/.bashrc

sudo mkdir /mnt/ImageFactData
#sudo sshfs adhofer@ImageFactData.cs.univie.ac.at:/./dataset /mnt/ImageFactData
# echo 'umount /mnt/ImageFactData/' >> ~/.bashrc
# echo 'rm -rf /mnt/ImageFactData/*' >> ~/.bashrc
# echo 'rm -rf /mnt/ImageFactData/' >> ~/.bashrc
# echo 'mkdir /mnt/ImageFactData/' >> ~/.bashrc
#sudo sshfs -o allow_other adhofer@ImageFactData.cs.univie.ac.at:/./dataset /mnt/ImageFactData
echo <vm_pwd> | sudo sshfs -o allow_other -o password_stdin adhofer@ImageFactData.cs.univie.ac.at:/./dataset /mnt/ImageFactData
echo 'sudo sshfs -o allow_other -o password_stdin adhofer@ImageFactData.cs.univie.ac.at:/./dataset /mnt/ImageFactData <<< <vm_pwd>' >> ~/.bashrc
source ~/.bashrc

#sudo apt install nfs-common
#sudo apt install cifs-utils

wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
#reopen shell
conda create --name sg python=3.6
conda activate sg
python -V

conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch
echo 'export CUDA_HOME=/usr/local/cuda-9.0/' >> ~/.bashrc
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo sh ./cuda_9.0.176_384.81_linux-run --no-opengl-libs --toolkit --silent --override

#conda install -c conda-forge menpo opencv --file requirements.txt
conda install pip
git clone https://github.com/Adlbert/graph-rcnn.pytorch.git
cd graph-rcnn.pytorch 
python -m pip install -r requirements.txt
python -m pip install scipy


cd lib/scene_parser/rcnn
python setup.py build develop
cd ..
cd ..
#Train Object detection
python main.py --config-file configs/faster_rcnn_res101.yaml
#Evaluate
python main.py --config-file configs/faster_rcnn_res101.yaml --inference --instance 100
#python -m torch.distributed.launch --nproc_per_node=2 main.py --config-file configs/faster_rcnn_res101.yaml
#Train SG
python main.py --config-file configs/sgg_res101_step.yaml
 python main.py --config-file configs/sgg_res101_step.yaml --resume checkpoints/vg_benchmark_object/R-101-C4/sg_baseline_relpn_step_0/BatchSize_4/Base_LR_0.005/checkpoint_0013499.pth
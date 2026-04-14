
python tsne_visualization.py --checkpoint /home/maqw/SARVLM/SARVLM/logs/transferclip_wo_mstar/checkpoints/epoch_8.pt \
    --dataset mstar \
    --prefix 'transferclip' 

python tsne_visualization.py --checkpoint /home/maqw/SARVLM/SARVLM/logs/transferclip_wo_opensarship/checkpoints/epoch_8.pt \
    --dataset fusar \
    --prefix 'transferclip'

python tsne_visualization.py --checkpoint /home/maqw/SARVLM/SARVLM/logs/transferclip_wo_opensarship_and_mstar/checkpoints/epoch_8.pt \
    --dataset sar_vsa \
    --prefix 'transferclip'

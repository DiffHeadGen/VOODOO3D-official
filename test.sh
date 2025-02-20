# # Test lp3d
# python test_lp3d.py --source_root resources/images/sources \
#                     --config_path configs/lp3d.yml \
#                     --model_path pretrained_models/voodoo3d.pth \
#                     --save_root results/lp3d_test \
#                     --cam_batch_size 5

# Test voodoo3d
# python test_voodoo3d.py --source_root resources/images/sources \
#                     --driver_root resources/images/drivers \
#                     --config_path configs/voodoo3d.yml \
#                     --model_path pretrained_models/voodoo3d.pth \
#                     --save_root results/voodoo3d_test \

CUDA_VISIBLE_DEVICES=1 python test_voodoo3d.py --source_root /nas_data/home/ldy/repos/diffheadgen/expdata/data/VFHQ_testset/Clip+_HebIzK_LP4+P2+C1+F16589-16715/ori_imgs/000000.png \
                    --driver_root /nas_data/home/ldy/repos/diffheadgen/expdata/data/VFHQ_testset/Clip+_HebIzK_LP4+P2+C1+F16589-16715/ori_imgs \
                    --config_path configs/voodoo3d.yml \
                    --model_path pretrained_models/voodoo3d.pth \
                    --save_root results/test
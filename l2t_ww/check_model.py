from l2t_ww.models import resnet_ilsvrc
from l2t_ww.models import resnet_cifar as cresnet, vgg_cifar as cvgg


def check_model(opt):
    if opt.model.startswith('resnet'):
        if opt.dataset in ['cub200', 'indoor', 'stanford40', 'flowers102', 'dog', 'tinyimagenet','cifar100','cifar10',
            'lung_nodules','camelyon17','gastric','lidc',"gastric_resplit1","gastric_resplit2","gastric_resplit2","early_lung_nodules","Endometrial_Cancer"]:
       # if opt.dataset in ['cub200', 'indoor', 'stanford40', 'flowers102', 'dog', 'tinyimagenet','camelyon17']:
            ResNet = resnet_ilsvrc.__dict__[opt.model]#(pretrained=True)
            model = ResNet(num_classes=opt.num_classes,mhsa=opt.target_mhsa,dropout_p=opt.dropout_p,resolution=(opt.input_shape,opt.input_shape),open_perturbe=opt.open_perturbe)
        else:
            ResNet = cresnet.__dict__[opt.model]
            model = ResNet(num_classes=opt.num_classes)

        return model

    elif opt.model.startswith('vgg'):
        VGG = cvgg.__dict__[opt.model]
        model = VGG(num_classes=opt.num_classes)

        return model
    elif opt.model.startswith('UNet'):
        Unet = resnet_ilsvrc.__dict__[opt.model]  # (pretrained=True)
        model = Unet(out_channels=opt.num_classes)

        return model
    else:
        raise Exception('Unknown model')

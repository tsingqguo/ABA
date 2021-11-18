
COLOR = ((1, 0, 0),
         (0, 1, 0),
         (1, 0, 1),
         (1, 1, 0),
         (0  , 162/255, 232/255),
         (0.5, 0.5, 0.5),
         (0, 0, 1),
         (0, 1, 1),
         (136/255, 0  , 21/255),
         (255/255, 127/255, 39/255),
         (0, 0, 0))

LINE_STYLE = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-']

MARKER_STYLE = ['o', 'v', '<', '*', 'D', 'x', '.', 'x', '<', '.']

def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/workspace/ABA/debug/{}.png'.format(name), save_im)
FNT_WQYZENHEI = '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc'

global font, texts, textColor
font = cfgReadFont(FNT_WQYZENHEI, 16)
texts, textColor = '#', color('#66ccff')

global graphSizef, spaceTextHV, paddingLR, paddingUB
graphSizef = lambda insz: insz
spaceTextHV = (2, 2)
paddingLR, paddingUB = (3,3), (6,6)

global drawTextq, colorAverage
pixelSampler = PixelSample.midCenter

backgroundPixelq = distanceBoundq(color('#FFFFFF'), 10, distancef=sumDistance)
drawTextq = compose(op_infix('>', percentage(50)),
  pixelRatio(backgroundPixelq, sampler=pixelSampler))

colorAverage = compose(multiAverage, pixelSampler)


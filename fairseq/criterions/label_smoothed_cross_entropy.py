# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import pytorch_lightning
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from scipy.spatial import distance as dist

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, EP, DP):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.DP = float(DP)
        self.EP = float(EP)


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--DP', default=0.0, metavar='DP', help='percentage of alignment loss to use as regularizer on decoder')
        parser.add_argument('--EP', default=0.0, metavar='EP', help='percentage of alignment loss to use as regularizer on encoder')

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        #GN: This needs to be fixed!
        #print("INPUT: ", sample)
        network_output = net_output[0]
        encoder_output = net_output[1][0].permute(1,0,-1)
        #print("OUTPUT: ", encoder_output.encoder_out[0].size())
        #print("OUTPUT2: ", encoder_output.encoder_out.size())

        #blah = encoder_output[0].permute(1,0,-1)
        #print("OUTPUT3: ", blah.size())
        #print("OUTPUT4: ", blah)

        lprobs = model.get_normalized_probs(network_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        decipherProbs = model.get_normalized_probs(network_output, log_probs=True)
        target = model.get_targets(sample, network_output).view(-1, 1)
        #print("TARGET: ", model.get_targets(sample,network_output).size())
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        #print("LOSS: ", loss)
        #print("NLL: ", nll_loss)
        #print("LPROBS: ", lprobs2.size())
        #print("TARGET: ", target.size())
        #print("SAMPLE: ", sample['target'].size())
        if torch.cuda.is_available():
            distanceLoss = torch.cuda.FloatTensor([0.0])
        
        else:
            distanceLoss = torch.FloatTensor([0.0])
          
        cos = torch.nn.CosineSimilarity(dim=1)
        #similarity = cos(encoder_output.encoder_out[0], encoder_output.encoder_out[0])
        #distance = encoder_output.encoder_out.size()[1] - similarity.sum()

        #print("SIM: ", similarity)
        #print("DIST: ", distance)
        #print("FULL TARG: ", sample['target'].size())
        for i in range(0, sample['target'].size()[0]):
            for j in range(i+1, sample['target'].size()[0]):
                targ1 = sample['target'][i]
                targ2 = sample['target'][j]
                if(not (torch.equal(targ1, targ2))):
                    continue
                #for k in range(encoder_output.encoder_out.size(0)):
                #    
                #    distance = 0.0         
                '''print("TARG1: ", targ1.size())
                print("TARG2: ", targ2.size())
                print("I: ", i)
                print("J: ", j)
                print("ENC1: ", encoder_output.encoder_out.size())
                print("ENC2: ", encoder_output.encoder_out[k].size())
                print("ENC3: ", encoder_output.encoder_out[k][0])
                print("ENC4: ", encoder_output.encoder_out[k][1])
                print("ENC5: ", decipherProbs[0])
                print("ENC6: ", decipherProbs[1])
                print("ENC7: ", decipherProbs[0].size())
                print("ENC8: ", decipherProbs[1].size())
                print("ENC9: ", encoder_output.encoder_out[k][0].size())
                print("ENC10: ", encoder_output.encoder_out[k][1].size())
                '''

                    #GN: This is the post-decoder version:
                    #
                similarityD = cos(decipherProbs[i], decipherProbs[j])
                distanceD = decipherProbs.size()[1] - similarityD.sum()
                    #
                    #GN: This is the pre-decoder version:
                similarityE = cos(encoder_output[i], encoder_output[j])
                        #print("SIM: ", similarity)
                distanceE = encoder_output.size()[1] - similarityE.sum(); #encoder_output.encoder_out.size()[0] - similarity.sum()
                    #                    
                    #print("TOTAL DISTANCE: ", distance)
                #similarity = cos(decipherProbs[i], decipherProbs[j])
                #distance = decipherProbs.size()[1] - similarity.sum()

                distance = (self.DP * distanceD) + (self.EP * distanceE) #* jaccard
                distanceLoss = distanceLoss.add(distance)
                #print("TARGETS: ", sample['target'])
                #print("SAMPLE: ", sample)
                #if(targDist[0] != 0.0):
                #    print("TARGETS: ", targ1, " ", targ2)
                #    print("TARGET: ", targDist)
                #    print("Union: ", union)
                #    print("Intersection: ", intersection)
        
        '''for i in range(0, sample['target'].size()[0]):
            for j in range(i+1, sample['target'].size()[0]):
                targ1 = sample['target'][i]
                targ2 = sample['target'][j]
                distance = 0.0         
                print("TARG1: ", targ1.size())
                print("TARG2: ", targ2.size())
                print("I: ", i)
                print("J: ", j)
                print("ENC1: ", encoder_output.encoder_out.size())
                
                #print("TARG1: ", targ1)
                #print("TARG2: ", targ2)
                
                #intersection = (targ1 & targ2).float()
                #uniq1 = targ1.unique()
                #uniq2 = targ2.unique()
        #        combined = torch.cat((targ1, targ2))
        #        uniques, counts = combined.unique(return_counts=True)
        #        difference = uniques[counts == 1]
        #        intersection = uniques[counts > 1]
        #        int_size = intersection.size()[0]
        #        union_size = int_size + difference.size()[0]
        #        jaccard = int_size / union_size
                #print("INTERSECTION: ", intersection)
                #print("SIZE: ", int_size)
                #print("DIFF: ", difference)
                #print("SIZE: ", union_size)
                
                #intersection = (targ1 == targ2).float().sum()
                #iou = pytorch_lightning.metrics.functional.classification.iou(targ1, targ2, reduction="none")

                #union = (targ1 != targ2).float().sum()
                
                #if torch.cuda.is_available():
                #    indices = torch.ones_like(targ1, dtype = torch.uint8, device = 'cuda')
                #    indicesU = torch.ones_like(targ1, dtype = torch.uint8, device = 'cuda')
                #else:
                #    indices = torch.ones_like(targ1, dtype = torch.uint8, device = 'cpu')
                #    indicesU = torch.ones_like(targ1, dtype = torch.uint8, device = 'cpu')
                     
                #for elem in targ2:
                #    indices = indices | (targ1 == elem)  
                #    indicesU = indicesU & (targ1 != elem)


                #intersection = torch.unique(targ1[indices])  
                #union = torch.unique(targ1[indicesU])        
                #print("UNION: ", union.size())        
                #print("INTERSECTION: ", intersection.size())        
                #print("Union: ", union)
                
                #union = {}
                #for k in range(len(targ1)):
                #    if(targ1[k].item() not in union):
                #        union[targ1[k].item()] = 'A'
                #intersection = 0
                #for k in range(len(targ2)):
                #    if(targ2[k].item() not in union):
                #        union[targ2[k].item()] = 'B'
                #    elif union[targ2[k].item()] == 'A':
                #        union[targ2[k].item()] = 'Both'
                #        intersection += 1
                #if torch.cuda.is_available():
                #    targDist = torch.cuda.FloatTensor([0.0])
                #else:
                #    targDist = torch.FloatTensor([0.0])
                #targDist = targDist.add(intersection.size()[0])
                #targDist = targDist.div(intersection.size()[0] + union.size()[0])
                #targDist = intersection.size()[0] / (intersection.size()[0] + union.size()[0]) #intersection / len(union)
                if(torch.equal(targ1, targ2)):

                    #GN: This is the post-decoder version:
                    #
                    #similarity = cos(decipherProbs[i], decipherProbs[j])
                    #distance = decipherProbs.size()[1] - similarity.sum()
                    #
                    #GN: This is the pre-decoder version:
                    similarity = cos(encoder_output.encoder_out[i], encoder_output.encoder_out[j])
                    distance = encoder_output.encoder_out.size()[1] - similarity.sum()
                    #                    

                    distance = self.alignLoss * distance #* jaccard
                    distanceLoss = distanceLoss.add(distance)
                #print("TARGETS: ", sample['target'])
                #print("SAMPLE: ", sample)
                #if(targDist[0] != 0.0):
                #    print("TARGETS: ", targ1, " ", targ2)
                #    print("TARGET: ", targDist)
                #    print("Union: ", union)
                #    print("Intersection: ", intersection)
        '''
        loss += distanceLoss.squeeze()
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

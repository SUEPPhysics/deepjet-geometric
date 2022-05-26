import argparse
import h5py
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage
from matplotlib.ticker import FixedLocator as Locator
import matplotlib.font_manager
from sklearn.metrics import precision_recall_curve


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                    '{0} is not a readable directory'.format(prospective_dir))


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid file'.format(prospective_file))
        else:
            setattr(namespace, self.dest, prospective_file)

class Plotting():

    def __init__(self, save_dir='./plots'):

        self.save_dir = save_dir

    def average_precision_score(self, recall, precision):

        def find_nearest(array, value):
            if array[1] < (value - .01):
                return None
            idx = (np.abs(array - value)).argmin()
            return idx

        recall[0] = 1.
        precision[0] = 0.

        cum_ap = 0.
        for i in [x / 10.0 for x in range(0, 11)]:
            x = find_nearest(recall, i)
            if x:
                cum_ap += precision[x]
        return cum_ap/11.

    def draw_loss(self,
                  data_train,
                  data_val,
                  name='',
                  keys=['Model 1', 'Model 2', 'Disco']):
        """Plots the training and validation loss"""

        fig, ax = plt.subplots()
        plt.xlabel("Epoch", horizontalalignment='right', x=1.0)
        plt.ylabel("Loss", horizontalalignment='right', y=1.0)
        plt.yscale("log")

        cmap = plt.cm.jet(np.linspace(0, 1, len(keys)+1))
        for train, val, key, color in zip(data_train, data_val, keys, cmap):
            plt.plot(train,
                     linestyle='solid',
                     color=color,
                     label=key)
            plt.plot(val,
                     linestyle='dashed',
                     color=color)

        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout()
        plt.savefig('{}/loss-{}'.format(self.save_dir, name), bbox_inches='tight')
        plt.close(fig)
        
    def draw_metrics(self, eventMetrics, eventMetricsVal, metricLabels, name):
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch", horizontalalignment='right', x=1.0)
        ax.set_ylabel("Metric Value", horizontalalignment='right', y=1.0)
        
        cmap = plt.cm.plasma(np.linspace(0, 1, len(metricLabels)+1))
        for metric, vmetric, n, c in zip(eventMetrics, eventMetricsVal, metricLabels, cmap):
            ax.plot(metric,
                    linestyle='solid',
                    color=c,
                    label=n)
            ax.plot(vmetric,
                    linestyle='dashed',
                    color=c)
            
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        fig.savefig('{}/metrics-{}'.format(self.save_dir, name))
        plt.close(fig)
            
#     def draw_precision_recall(self,
#                               results,
#                               jet_names):
#         """Plots the precision recall curve"""

#         def find_nearest(array, value):
#             if array[1] < (value - .01):
#                 return None
#             idx = (np.abs(array - value)).argmin()
#             return idx

#         fig, ax = plt.subplots()
#         results_ap, results_pr3, results_pr5 = [], [], []
#         for i, results in enumerate([results_base,
#                                      results_fpn,
#                                      results_twn,
#                                      results_int8]):
            
#             if results is None: 
#                 results_ap.append(None)
#                 results_pr3.append(None)
#                 results_pr5.append(None)
#                 continue
            
#             name = self.legend[i]
#             scores, truths = [], []
#             tmp_ap, tmp_pr3, tmp_pr5 = [], [], []
#             for j, jet in enumerate(jet_names):
#                 truth = results[j][:, 4].numpy()
#                 score = results[j][:, 3].numpy()
#                 truths = np.concatenate((truths, truth), axis=None)
#                 scores = np.concatenate((scores, score), axis=None)
#                 precision, recall, _ = precision_recall_curve(truth, score)
#                 ap = self.average_precision_score(recall, precision)
#                 tmp_ap.append(ap)
#                 x = find_nearest(recall, 0.3)
#                 if x is None:
#                     tmp_pr3.append(np.nan)
#                 else:
#                     tmp_pr3.append(precision[x])
#                 x = find_nearest(recall, 0.5)
#                 if x is None:
#                     tmp_pr5.append(np.nan)
#                 else:
#                     tmp_pr5.append(precision[x])
                    
                    
#                 print(r'{0}, AP: {1:.3f}'.format(name, ap))
#                 label = '{0}: {1} jets, AP: {2:.3f}'.format(name, jet, ap)
#                 plt.plot(recall[1:],
#                      precision[1:],
#                      linestyle=self.line_styles[i],
#                      linewidth=.5,
#                      markersize=0,
#                      color=self.colors[j],
#                      label=label)

#             results_ap.append(tmp_ap)
#             results_pr3.append(tmp_pr3)
#             results_pr5.append(tmp_pr5)

#         plt.xlabel(r'Recall (TPR)', horizontalalignment='right', x=1.0)
#         plt.ylabel(r'Precision (PPV)', horizontalalignment='right', y=1.0)
#         plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
#         plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
#         ax.legend(loc='lower right')
#         fig.savefig('{}/Precision-Recall-Curve'.format(self.save_dir))
#         plt.close(fig)
#         return results_ap, results_pr3, results_pr5

    def draw_precision_recall(self, 
                            y_preds,
                            y_test,
                            name='',
                            keys=['Model 1', 'Model 2']):
        
        fig = plt.figure()
        ax = fig.subplots()
        cmap = plt.cm.jet(np.linspace(0, 1, len(keys)+1))
        
        if len(y_preds) == len(y_test):
            pre, rec, _ = precision_recall_curve(y_test, y_preds)
            ax.plot(rec, pre, label=keys[0], color=cmap[0])
        else:   
            for key, y_pred, color in zip(keys, y_preds, cmap):
                pre, rec, _ = precision_recall_curve(y_test, y_pred)
                ax.plot(rec, pre, label=key, color=color)

        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title("Precision-Recall")
        ax.legend()
        fig.tight_layout()
        fig.savefig('{}/precision-recall-curve-{}'.format(self.save_dir, name), bbox_inches='tight')
        plt.close(fig)

    def draw_disco(self, results, 
                   name='', 
                   keys=['Model 1', 'Model 2'],
                   xlim=[0,1],
                   ylim=[0,1]):
        """ Plots results for double disco """
                
        with open('{}/disco_results.npy'.format(self.save_dir), 'wb') as f:
            np.save(f, results)
        
        fig = plt.figure(figsize=(10, 5))
        ax, ax2 = fig.subplots(1,2)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax2.set_xlabel(keys[0])
        ax2.set_ylabel(keys[1])
        ax.set_title("SUEP")
        ax2.set_title("QCD")
        
        suep_1 = results[0][results[2] == 1]
        suep_2 = results[1][results[2] == 1]
        qcd_1 = results[0][results[2] == 0]
        qcd_2 = results[1][results[2] == 0]
        
        ax.hist2d(suep_1, suep_2, bins=[np.linspace(xlim[0],xlim[1],40), np.linspace(ylim[0],ylim[1],40)], label='SUEP')
        ax2.hist2d(qcd_1, qcd_2, bins=[np.linspace(xlim[0],xlim[1],40), np.linspace(ylim[0],ylim[1],40)], label='QCD')
        ax.set_xlim(xlim[0],xlim[1])
        ax2.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        ax2.set_ylim(ylim[0],ylim[1])
        
        fig.tight_layout()
        fig.suptitle("Model "+ name)
        
        fig.savefig('{}/disco-reults-{}.png'.format(self.save_dir, name))
        plt.close(fig)

    def draw_precision_details(self, gt, fpn, twn, int8, jet_names, nbins=11):
        """Plots the precision histogram at fixed recall"""
        ylabel = 'PPV@R={}'.format(self.ref_recall)
        xlabels = [r'$\eta$', r'$\phi$ [°]', r'$p_T^{SSD}$ [GeV/s]']
        scales = [0.4, 0.5, 20]
        idxs = [0, 1, 5]
        ax_m = [6, 360, 1]
        ax_s = [3, 0, 0]

        for x, jet_name in enumerate(jet_names):
            fig, axs = plt.subplots(3, 3, figsize=(10.5, 5.4))
            for row, result in enumerate([fpn, twn, int8]):
                if result is None: continue
                for column, (i, l, ax_mul, ax_sub) in enumerate(
                        zip(idxs, xlabels, ax_m, ax_s)):

                    ax = axs[row][column]

                    if row == 2:
                        ax.set_xlabel(l, horizontalalignment='right', x=1.0)
                    if column == 0:
                        ax.set_ylabel(ylabel,
                                      horizontalalignment='right',
                                      y=1.0)

                    # Fix binning across classes
                    if i == 5 and gt is not None:
                        pt = gt[gt[:, 0] == x+1][:, 1].numpy()
                        min_pt, max_pt = np.min(pt), np.max(pt)
                        binning = np.logspace(np.log10(min_pt),
                                              np.log10(max_pt),
                                              nbins)[1:]
                        ax.set_xscale("log")
                        ax.set_xlim([min_pt, 1.1*max_pt])
                        ax.set_ylim([0, 1.1])
                    else:
                        binning = np.linspace(0, 1, nbins)[1:]
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1.1])

                    score = result[x][:, 3].numpy()
                    truth = result[x][:, 4].numpy()
                    values = result[x][:, i].numpy()
                    bmin, v = 0, []
                    for bmax in binning:
                        if binning[-1] == bmax:
                            mask = (values > bmin)
                        else:
                            mask = (values > bmin) & (values <= bmax)
                        s, t = score[mask], truth[mask]
                        if len(s) and np.sum(t):
                            p, r, _ = precision_recall_curve(t, s)
                            tmp = p[(np.abs(r - self.ref_recall)).argmin()]
                            v.append(np.round(tmp, 2))
                        else:
                            v.append(np.nan)
                        bmin = bmax

                    if i == 5:
                        xvalues = binning
                    else:
                        xvalues = binning-binning[0]/2

                    ax.plot(xvalues,
                            v,
                            color=self.colors[0],
                            marker=self.markers[0],
                            linewidth=0)

                    if i == 0:
                        ticks = ax.get_xticks()*ax_mul-ax_sub
                        ticks = np.round_(ticks, decimals=2)
                        ax.xaxis.set_major_locator(Locator(ax.get_xticks()))
                        ax.set_xticklabels(ticks)
                    if i == 1:
                        ticks = ax.get_xticks()*ax_mul-ax_sub
                        ticks = ticks.astype(np.int32)
                        ax.xaxis.set_major_locator(Locator(ax.get_xticks()))
                        ax.set_xticklabels(ticks)
                    plt.setp(ax.get_yticklabels(), visible=column == 0)
                    plt.setp(ax.get_xticklabels(), visible=row == 2)

            plt.savefig('{}/Precision-{}'.format(self.save_dir, jet_name))
            plt.close(fig)

    def draw_loc_delta(self, base, fpn, twn, int8, jet_names, nbins=11):
        """Plots the localization and regression error"""
        xlabel = r'$p_T^{GEN}$ [GeV/s]'
        ylabels = [r'$\eta-\eta^{GEN}$',
                   r'$\phi-\phi^{GEN}$ [rad]',
                   r'$|\frac{p_T}{p_T^{GEN}}|$']
        scales = [0.4, 0.5, 20]
        idxs = [2, 3, 4]
        
        for x, jet_name in enumerate(jet_names):
            fig, axs = plt.subplots(3, 4, figsize=(14, 5.4))
            for row, (idx, ylabel, s) in enumerate(zip(idxs, ylabels, scales)):
                # Fix binning across classes
                if base is not None:
                    pt = base[base[:, 0] == x+1][:, 1].numpy()
                    min_pt, max_pt = np.min(pt), np.max(pt)
                    binning = np.logspace(np.log10(min_pt),
                                          np.log10(max_pt),
                                          nbins)[1:]
                else:
                    pt = fpn[fpn[:, 0] == x+1][:, 1].numpy()
                    min_pt, max_pt = np.min(pt), np.max(pt)
                    binning = np.logspace(np.log10(min_pt),
                                          np.log10(max_pt),
                                          nbins)[1:]
                for column, results in enumerate([base, fpn, twn, int8]):
                    
                    if results is None: continue
                    ax = axs[row][column]

                    if row == 2:
                        ax.set_xlabel(xlabel,
                                      horizontalalignment='right',
                                      x=1.0)
                    if column == 0:
                        ax.set_ylabel(ylabel,
                                      horizontalalignment='right',
                                      y=1.0)

                    cls = results[results[:, 0] == x+1].numpy()
                    bmin, v, e = 0, [], []
                    for bmax in binning:
                        b = cls[(cls[:, 1] > bmin) & (cls[:, 1] <= bmax)]
                        if row == 2:
                            absb = np.abs(b[:, idx])
                        else:
                            absb = b[:, idx]
                        if len(absb):
                            v.append(np.mean(absb))
                            e.append(np.std(absb))
                        else:
                            v.append(np.nan)
                            e.append(np.nan)
                        bmin = bmax

                    ax.errorbar(binning,
                                v,
                                yerr=e,
                                ecolor=self.colors[0],
                                color=self.colors[0],
                                marker=self.markers[0],
                                capsize=2,
                                elinewidth=0.5,
                                linewidth=0)

                    if row == 2:
                        ax.set_ylim([0, s])
                        ax.set_yscale("symlog")
                    else:
                        ax.set_ylim([-s, s])

                    ax.set_xlim([min_pt, max_pt*1.2])
                    ax.set_xscale("log")
                    plt.setp(ax.get_yticklabels(), visible=column == 0)
                    plt.setp(ax.get_xticklabels(), visible=row == 2)
            plt.savefig('%s/Delta-%s' % (self.save_dir, jet_name))
            plt.close(fig)

    def draw_barchart(self, x, y1, y2, label, ylabel,
                      xlabel='Batch size [events]',
                      save_name='inference'):
        """Plots errobars as a function of batch size"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.625))

        width = 0.11
        groups = np.arange(len(x))

        ax1.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
        ax1.set_ylabel(ylabel[0], horizontalalignment='right', y=1.0)
        ax1.bar(groups - 0.36, y1[0], label=label[0], width=width)
        ax1.bar(groups - 0.24, y1[1], label=label[1], width=width)
        ax1.bar(groups - 0.12, y1[2], label=label[2], width=width)
        ax1.bar(groups + 0.00, y1[3], label=label[3], width=width)
        ax1.bar(groups + 0.12, y1[4], label=label[4], width=width)
        ax1.bar(groups + 0.24, y1[5], label=label[5], width=width)
        ax1.bar(groups + 0.36, y1[6], label=label[6], width=width)
        ax1.set_xticks(groups)
        ax1.set_xticklabels(x)
        ax1.set_yscale('log')

        ax2.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
        ax2.set_ylabel(ylabel[1], horizontalalignment='right', y=1.0)
        ax2.bar(groups - 0.36, y2[0], label=label[0], width=width)
        ax2.bar(groups - 0.24, y2[1], label=label[1], width=width)
        ax2.bar(groups - 0.12, y2[2], label=label[2], width=width)
        ax2.bar(groups + 0.00, y2[3], label=label[3], width=width)
        ax2.bar(groups + 0.12, y2[4], label=label[4], width=width)
        ax2.bar(groups + 0.24, y2[5], label=label[5], width=width)
        ax2.bar(groups + 0.36, y2[6], label=label[6], width=width)
        ax2.set_xticks(groups)
        ax2.set_xticklabels(x)
        ax2.set_yscale('log')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles,
                   labels,
                   bbox_to_anchor=(0.1, 1.02),
                   loc='upper left',
                   ncol=4,
                   fontsize=7)
        fig.savefig('{}/{}'.format(self.save_dir, save_name))
        plt.close(fig)


class GetResources():

    def __init__(self, net, dummy_input):
        self.net = net
        self.dummy_input = dummy_input

    def zero_ops(self, m, x, y):
        m.total_ops += torch.DoubleTensor([int(0)]).cuda()

    def count_bn(self, m, x, y):
        x = x[0]
        nelements = 2 * x.numel()
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

    def count_conv(self, m, x, y):
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
        total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)
        m.total_ops += torch.DoubleTensor([int(total_ops)]).cuda()

    def count_prelu(self, m, x, y):
        x = x[0]
        nelements = x.numel()
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

    def profile(self):
        handler_collection = {}
        types_collection = set()

        register_hooks = {
            nn.Conv2d: self.count_conv,
            nn.BatchNorm2d: self.count_bn,
            nn.PReLU: self.count_prelu,
            nn.AvgPool2d: self.zero_ops
        }

        def add_hooks(m: nn.Module):
            m.register_buffer('total_ops', torch.zeros(1))
            m_type = type(m)

            fn = None
            if m_type in register_hooks:
                fn = register_hooks[m_type]
            if fn is not None:
                handler_collection[m] = (m.register_forward_hook(fn))

            types_collection.add(m_type)

        def dfs_count(module: nn.Module, prefix="\t"):
            total_ops = 0
            for m in module.children():
                if m in handler_collection and not isinstance(
                          m, (nn.Sequential, nn.ModuleList)):
                    ops = m.total_ops.item()
                else:
                    ops = dfs_count(m, prefix=prefix + "\t")
                total_ops += ops
            return total_ops
        self.net.eval()
        self.net.apply(add_hooks)
        with torch.no_grad():
            self.net(self.dummy_input)
        total_ops = dfs_count(self.net)

        return total_ops


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = list(transposed_data[1])
    if len(transposed_data) < 3:
        return inp, tgt
    if len(transposed_data) < 4:
        slr = list(transposed_data[2])
        return inp, tgt, slr
    bsl = list(transposed_data[2])
    slr = list(transposed_data[3])
    return inp, tgt, bsl, slr

def set_logging(name, filename, verbose):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(filename)
    ch = logging.StreamHandler()

    logger.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    if verbose:
        ch.setLevel(logging.INFO)

    f = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                          datefmt='%m/%d/%Y %I:%M')
    fh.setFormatter(f)
    ch.setFormatter(f)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

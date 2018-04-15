from util import *

class StruOptim(object):
    def __init__(self,creator,reader,map,valid_batch,valid_iter,input,start,end):
        numpy.seterr(over='raise')
        self.network = creator
        self.reader = reader
        self.batch_size = valid_batch
        self.mapping = map
        self.loop = valid_iter
        self.input = input
        self.start = start
        self.end = end
        self.hist_para = []
        self.hist_perform = []
        self.num_para = len(self.start)
    def _node(self,para_list):
        n = Node(para_list, self.network, self.reader, self.mapping, self.batch_size, self.loop, self.input)
        return [n.accuracy, n.time]

    def _init(self,samples):
        sample_hist = []
        samples = min(samples, numpy.min(numpy.array(self.end) - numpy.array(self.start)))
        for i in range(self.num_para):
            _range = range(self.start[i], self.end[i])
            inc = random.sample(_range, samples)
            sample_hist.append(inc)
        for i in range(samples):
            para = []
            for j in range(self.num_para):
                para.append(sample_hist[j][i])
            self.hist_para.append(para)
        for para in self.hist_para:
            self.hist_perform.append(self._node(para))

    def _gen_cut(self,des,ori,step=3):
        group = []
        for i in xrange(self.num_para):
            if des[i] != ori[i]:
                if des[i] - ori[i] == 1:
                    break
                _step = min(step, des[i] - ori[i] - 1)
                _range = range(des[i] - ori[i] - 1)
                inc = random.sample(_range, _step)
                for s in inc:
                    new = list(ori)
                    new[i] += s + 1
                    group.append(new)
                break
        return group

    def _next_gen_linear(self,para_list,step=16):
        group = []
        for i in xrange(self.num_para):
            new_list = list(para_list)
            if len(step) == 1:
                new_list[i] += step
            else:
                new_list[i] += step[i]
            if (numpy.array(new_list) <= numpy.array(self.end)).all():
                group.append(new_list)
        return group

    def _gird_mu(self,count=10000):
        a = []
        for i in range(self.num_para):
            a.append(random.sample(range(self.start[i], self.end[i] + 1), int(numpy.power(count, 1.0 / self.num_para))))
        mu = []
        for point in itertools.product(*a):
            mup,mut,sp,st,s = get_mu(list(point),self.hist_para,self.hist_perform)
            mu.append([mup,mut])
        mup = numpy.percentile(numpy.array(mu)[:, 0], 99)
        mut = numpy.percentile(numpy.array(mu)[:, 1], 1)
        return max(0,mup), -min(0,mut),s,sp,st

    def _mu_trans(self,ori,performance,gird_count = 10000):
        up, ut, s, sp, st = self._gird_mu(gird_count)
        up = min(100,up)
        ut = min(100,ut)
        if up >0.01:
            rp = 1 - numpy.exp(up * (sp.transform(numpy.array(ori[0]).reshape(1,1))
                                 - sp.transform(performance[:, 0].reshape(-1,1))))
        else:
            rp = performance[:, 0].reshape(-1,1) - ori[0]
        if ut > 0.01:
            rt = numpy.exp(ut * (st.transform(performance[:, 1].reshape(-1,1))
                             - st.transform(numpy.array(ori[1]).reshape(1,1)))) - 1
        else:
            rt = performance[:, 1].reshape(-1,1) - ori[1]
        #print up,ut
        return rp / rt

    def _loop_body(self,current,forward_step = 16, backward_step = 3,gird_count = 10000):
        t_eps = 1e-6
        self.hist_para.append(current)
        ori = self._node(current)
        self.hist_perform.append(ori)
        group = self._next_gen_linear(current,forward_step)
        if len(group) == 0:
            return current,[-1,0]
        performance = []
        for new_para in group:
            vec = self._node(new_para)
            if vec[0] < ori[0]: vec[0] = ori[0]
            if vec[1] < ori[1]: vec[1] = ori[1] + t_eps
            performance.append(vec)
            self.hist_para.append(new_para)
            self.hist_perform.append(vec)
        arr_performance = numpy.array(performance)
        p_idx = numpy.argmax(arr_performance[:, 0])
        ratio_score = self._mu_trans(ori,arr_performance,gird_count)
        t_idx = numpy.argmax(ratio_score)

        new_group = self._gen_cut(group[p_idx], current,step = backward_step)
        if p_idx != t_idx: new_group += self._gen_cut(group[t_idx], current,step = backward_step)
        for new_para in new_group:
            vec = self._node(new_para)
            if vec[0] < ori[0]: vec[0] = ori[0]
            if vec[1] < ori[1]: vec[1] = ori[1] + t_eps
            performance.append(vec)
            group.append(new_para)
            self.hist_perform.append(vec)
            self.hist_para.append(new_para)
        arr_performance = numpy.array(performance)
        ratio_score = self._mu_trans(ori, arr_performance, gird_count)
        idx = numpy.argmax(ratio_score)
        #print ratio_score.transpose(),idx
        return group[idx],arr_performance[idx]

    def start_optim(self,base_line,bound_style,init_samples = 20, forward_step = 16,backward_step = 3,girds = 10000):
        print 'Initializing the searching space...'
        self._init(init_samples)
        print 'Finished'
        print 'Checking baseline performance...'
        [perform_bound,time_bound] = self._node(base_line)
        print perform_bound,time_bound
        tol = 0
        if bound_style == 'time':
            current = self.start
            perform = [0,0]
            while  perform[0] >= 0:
                if perform[1] < time_bound:
                    tol = 0
                else:
                    tol = tol + 1
                if tol > 5:
                    break
                current,perform = self._loop_body(current, forward_step, backward_step, girds)
                print current,perform

        elif bound_style == 'perform':
            current = self.start
            perform = [0, 0]
            while 0 <= perform[0] < perform_bound:
                current, perform = self._loop_body(current, forward_step, backward_step, girds)
                print current, perform
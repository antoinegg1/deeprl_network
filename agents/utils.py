import numpy as np
import tensorflow.compat.v1 as tf

"""
initializers
"""
DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'

def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2: # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4): # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
    def _norm_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            n_in = shape[0]
        elif (len(shape) == 3) or (len(shape) == 4):
            n_in = np.prod(shape[:-1])
        a = np.random.standard_normal(shape)
        if mode == 'fan_in':
            n = n_in
        elif mode == 'fan_out':
            n = shape[-1]
        elif mode == 'fan_avg':
            n = 0.5 * (n_in + shape[-1])
        return (scale * a / np.sqrt(n)).astype(np.float32)

DEFAULT_METHOD = ortho_init
"""
layers
"""
def conv(x, scope, n_out, f_size, stride=1, pad='VALID', f_size_w=None, act=tf.nn.relu,
         conv_dim=1, init_scale=DEFAULT_SCALE, init_mode=None, init_method=DEFAULT_METHOD):
    with tf.variable_scope(scope):
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        if conv_dim == 1:
            n_c = x.shape[2].value
            w = tf.get_variable("w", [f_size, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv1d(x, w, stride=stride, padding=pad) + b
        elif conv_dim == 2:
            n_c = x.shape[3].value
            if f_size_w is None:
                f_size_w = f_size
            w = tf.get_variable("w", [f_size, f_size_w, n_c, n_out],
                                initializer=init_method(init_scale, init_mode))
            z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
        return act(z)


def fc(x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
       init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    with tf.variable_scope(scope):
        n_in = x.shape[1].value
        w = tf.get_variable("w", [n_in, n_out],
                            initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        if act==None:
            return z
        return act(z)


def batch_to_seq(x):
    n_step = x.shape[0].value
    if len(x.shape) == 1:
        x = tf.expand_dims(x, -1)
    return tf.split(axis=0, num_or_size_splits=n_step, value=x)


def seq_to_batch(x):
    return tf.concat(x, axis=0)


def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
         init_method=DEFAULT_METHOD):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = xs[0].shape[1].value
    n_out = s.shape[0] // 2
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [n_in, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        wh = tf.get_variable("wh", [n_out, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
    s = tf.expand_dims(s, 0)
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[ind] = h
    s = tf.concat(axis=1, values=[c, h])
    return seq_to_batch(xs), tf.squeeze(s)


def lstm_comm(xs, ps, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
              init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = int(s.shape[1] // 2)
    n_s = int(xs.shape[-1])
    n_a = int(ps.shape[-1])
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    w_fp = []
    b_fp = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    n_in_hid = 3*n_h
    for i in range(n_agent):
        n_m = int(np.sum(masks[i]))
        # n_in_hid = (n_m+1)*n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            w_fp.append(tf.get_variable("w_fp", [n_a*n_m, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_fp.append(tf.get_variable("b_fp", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_in_hid, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        # x = batch_to_seq(tf.squeeze(x, axis=0))
        # p = batch_to_seq(tf.squeeze(p, axis=0))
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        # for i in range(n_agent):
            # hi = tf.expand_dims(h[i], axis=0)
            # hxi = fc(xi, 'mfc_s_%d' % i, n_h, act=tf.nn.tanh)
            # hpi = fc(pi, 'mfc_p_%d' % i, n_h, act=tf.nn.tanh)
            # si = tf.concat([hi, hxi, hpi], axis=1)
            # mi = fc(hi, 'mfc_%d' % i, n_h)
            # out_m.append(mi)
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
            pi = tf.expand_dims(tf.reshape(tf.boolean_mask(p, masks[i]), [-1]), axis=0)
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            hpi = tf.nn.relu(tf.matmul(pi, w_fp[i]) + b_fp[i])
            hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
            si = tf.concat([hxi, hpi, hmi], axis=1)
            # si = tf.concat([hxi, hmi], axis=1)
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_comm_hetero(xs, ps, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                     init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    w_fp = []
    b_fp = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    na_dim_ls = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        n_fp = 0
        na_dim = []
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            n_fp += n_a_ls[j]
            na_dim.append(n_a_ls[j])
            ns_dim.append(n_s_ls[j])
        na_dim_ls.append(na_dim)
        ns_dim_ls.append(ns_dim)
        n_m = len(ns_dim)
        if n_m:
            n_in_hid = 3*n_h
        else:
            n_in_hid = n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            if n_m:
                w_fp.append(tf.get_variable("w_fp", [n_fp, n_h],
                                            initializer=init_method(init_scale, init_mode)))
                b_fp.append(tf.get_variable("b_fp", [n_h],
                                            initializer=tf.constant_initializer(0.0)))
                w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_fp.append(None)
                b_fp.append(None)
                w_msg.append(None)
                b_msg.append(None)
            wx_hid.append(tf.get_variable("wx_hid", [n_in_hid, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            pi = []
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
                raw_pi = tf.boolean_mask(p, masks[i]) # n_n*n_a
                raw_xi = tf.boolean_mask(x, masks[i])
                # find the valid information based on each agent's s, a dim
                for j in range(n_m):
                    pi.append(tf.slice(raw_pi, [j, 0], [1, na_dim_ls[i][j]]))
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                hpi = tf.nn.relu(tf.matmul(tf.concat(pi, axis=1), w_fp[i]) + b_fp[i])
                hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
                si = tf.concat([hxi, hpi, hmi], axis=1)
            else:
                si = hxi
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_ic3(xs, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
             init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = int(xs.shape[-1])
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    for i in range(n_agent):
        n_m = int(np.sum(masks[i]))
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, done) in enumerate(zip(xs, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        out_h = []
        out_c = []
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.reduce_mean(tf.boolean_mask(out_m, masks[i]), axis=0, keepdims=True)
            # the state encoder in IC3 code is not consistent with that described in the paper.
            # Here we follow the impelmentation in the paper.
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            si = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i]) + tf.matmul(mi, w_msg[i]) + b_msg[i]
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_ic3_hetero(xs, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                    init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            ns_dim.append(n_s_ls[j])
        n_m = len(ns_dim)
        ns_dim_ls.append(ns_dim)
        with tf.variable_scope(scope + ('_%d' % i)):
            if n_m:
                w_msg.append(tf.get_variable("w_msg", [n_h, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_msg.append(None)
                b_msg.append(None)
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, done) in enumerate(zip(xs, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        out_h = []
        out_c = []
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.reduce_mean(tf.boolean_mask(out_m, masks[i]), axis=0, keepdims=True)
                raw_xi = tf.boolean_mask(x, masks[i])
                for j in range(n_m):
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            # the state encoder in IC3 code is not consistent with that described in the paper.
            # Here we follow the impelmentation in the paper.
            si = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                si = si + tf.matmul(mi, w_msg[i]) + b_msg[i]
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_dial(xs, ps, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
              init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = xs.shape[-1]
    n_a = ps.shape[-1]
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    for i in range(n_agent):
        n_m = np.sum(masks[i])
        # n_in_hid = (n_m+1)*n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        for i in range(n_agent):
            hi = tf.expand_dims(h[i], axis=0)
            mi = fc(hi, 'mfc_%d' % i, n_h)
            out_m.append(mi)
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
            ai = tf.one_hot(tf.expand_dims(tf.argmax(p[i]), axis=0), n_h)
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
            si = hxi + hmi + ai
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_dial_hetero(xs, ps, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                     init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            ns_dim.append(n_s_ls[j])
        n_m = len(ns_dim)
        ns_dim_ls.append(ns_dim)
        with tf.variable_scope(scope + ('_%d' % i)):
            if n_m:
                w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_msg.append(None)
                b_msg.append(None)
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        for i in range(n_agent):
            hi = tf.expand_dims(h[i], axis=0)
            mi = fc(hi, 'mfc_%d' % i, n_h)
            out_m.append(mi)
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
                ai = tf.one_hot(tf.expand_dims(tf.argmax(p[i]), axis=0), n_h)
                raw_xi = tf.boolean_mask(x, masks[i])
                for j in range(n_m):
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            si = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
                si = si + hmi + ai
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s

def lstm_ic3_gate(xs, dones, masks, s, scope,
                  init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
                  init_method=DEFAULT_METHOD):
    """IC3Net communication core (gating INSIDE the time loop).

    Parameters
    ----------
    xs : Tensor (N,T,n_s)   observations
    dones : Tensor (T,)     episode‑done flags
    masks : ndarray (N,N)   neighbourhood adjacency (0/1, no self‑loop)
    s : Tensor (N, 2*n_h)   concatenated [c,h] states
    scope : str             variable scope root

    Returns
    -------
    h_seq : Tensor (N,T,n_h) final hidden states sequence (after update)
    new_s : Tensor (N,2*n_h) concatenated next [c,h]
    g_seq : Tensor (N,T,1)   gate values (for sparsity loss)
    """
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = int(xs.shape[-1])

    # Re‑organize batch/seq dims to match tf.nn.rnn utilities
    xs = tf.transpose(xs, perm=[1, 0, 2])       # T,N,n_s
    xs = batch_to_seq(xs)                       # list(T) of (N,n_s)
    dones = batch_to_seq(dones)                 # list(T) of (1,)

    # ===== create/collect weights (one scope per agent) =========
    w_msg, b_msg, w_ob, b_ob = [], [], [], []
    wx_hid, wh_hid, b_hid    = [], [], []
    w_gate, b_gate           = [], []

    for i in range(n_agent):
        n_m = int(np.sum(masks[i]))
        with tf.variable_scope(scope + ('_%d' % i)):
            # original CommNet/IC3 weights
            w_msg.append(tf.get_variable('w_msg', [n_h, n_h],
                                          initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable('b_msg', [n_h],
                                          initializer=tf.zeros_initializer()))
            w_ob.append(tf.get_variable('w_ob', [n_s * (n_m + 1), n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable('b_ob', [n_h], initializer=tf.zeros_initializer()))
            wx_hid.append(tf.get_variable('wx_hid', [n_h, n_h * 4],
                                           initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable('wh_hid', [n_h, n_h * 4],
                                           initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable('b_hid', [n_h * 4], initializer=tf.zeros_initializer()))
            # --- gate weights ---
            w_gate.append(tf.get_variable('w_gate', [n_h, 1],
                                           initializer=init_method(init_scale, init_mode)))
            b_gate.append(tf.get_variable('b_gate', [1], initializer=tf.zeros_initializer()))

    # split initial state
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)   # (N,n_h) each

    g_time_list = []   # to collect gate values over time

    # ================ time‑step loop ============================
    for t, (x_t, done_t) in enumerate(zip(xs, dones)):
        # squeeze list element dims: x_t (N,n_s), done_t (1,)
        x_t = tf.squeeze(x_t, axis=0)
        done_t = tf.squeeze(done_t, axis=0)   # scalar 0/1

        # ---- compute gates & neighbour weighted messages (based on PREVIOUS h) ----
        g_list = []
        m_list = []
        for i in range(n_agent):
            h_i = tf.expand_dims(h[i], axis=0)                      # 1,n_h
            g_i = tf.sigmoid(tf.matmul(h_i, w_gate[i]) + b_gate[i]) # 1,1
            g_list.append(g_i)
            m_list.append(g_i * h_i)
        g_mat = tf.concat(g_list, axis=0)     # N,1
        m_mat = tf.concat(m_list, axis=0)     # N,n_h

        # make neighbour mask tensor once per step
        mask_tf = tf.constant(masks, dtype=tf.float32)             # N,N

        out_h, out_c = [], []
        g_time_list.append(tf.expand_dims(g_mat, axis=1))          # keep time dim

        # ========== agent update loop ==========
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)

            # reset on done
            ci = ci * (1 - done_t)
            hi = hi * (1 - done_t)

            # neighbour gated mean
            mi_num = tf.reduce_sum(tf.boolean_mask(m_mat, masks[i]), axis=0, keepdims=True)
            gi_den = tf.reduce_sum(tf.boolean_mask(g_mat, masks[i]), axis=0, keepdims=True) + 1e-6
            mi = mi_num / gi_den                                 # 1,n_h

            # state encoder (paper style)
            xi_nei = tf.expand_dims(tf.reshape(tf.boolean_mask(x_t, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x_t[i], axis=0), xi_nei], axis=1)  # 1,n_s*(n_m+1)
            si = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i]) + tf.matmul(mi, w_msg[i]) + b_msg[i]

            # LSTM cell update
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.sigmoid(ii); fi = tf.sigmoid(fi); oi = tf.sigmoid(oi); ui = tf.tanh(ui)
            ci = fi * ci + ii * ui
            hi = oi * tf.tanh(ci)

            out_h.append(hi)
            out_c.append(ci)

        # concat updated states
        c = tf.concat(out_c, axis=0)      # N,n_h
        h = tf.concat(out_h, axis=0)      # N,n_h
        xs[t] = tf.expand_dims(h, axis=0) # store sequence

    # post loop --------------------------------------------------
    s_new = tf.concat(axis=1, values=[c, h])            # N,2*n_h
    h_seq = seq_to_batch(xs)                            # T,N,n_h
    h_seq = tf.transpose(h_seq, perm=[1,0,2])           # N,T,n_h
    g_seq = tf.concat(g_time_list, axis=1)              # N,T,1
    return h_seq, s_new, g_seq

def lstm_ic3_gate_update(xs, dones, masks, s, scope,
                  init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
                  init_method=DEFAULT_METHOD,
                  k_gate=4,                    
                  use_layer_norm=False,      
                  dropout_p=0.1,             
                  use_attention=True):       
    n_agent = s.shape[0]
    n_h     = s.shape[1] // 2
    n_s     = int(xs.shape[-1])

    xs    = tf.transpose(xs, perm=[1, 0, 2])  # T,N,n_s
    xs    = batch_to_seq(xs)                  # list(T) of (N,n_s)
    dones = batch_to_seq(dones)               # list(T) of (1,)

    w_msg, b_msg, w_ob, b_ob = [], [], [], []
    wx_hid, wh_hid, b_hid    = [], [], []
    w_gate, b_gate, tau_list = [], [], []    

    for i in range(n_agent):
        n_m = int(np.sum(masks[i]))
        with tf.variable_scope(scope + f'_{i}'):
            w_msg.append(tf.get_variable('w_msg', [n_h, n_h],
                                          initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable('b_msg', [n_h], initializer=tf.zeros_initializer()))
            w_ob.append(tf.get_variable('w_ob', [n_s * (n_m + 1), n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable('b_ob', [n_h], initializer=tf.zeros_initializer()))
            wx_hid.append(tf.get_variable('wx_hid', [n_h, n_h * 4],
                                           initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable('wh_hid', [n_h, n_h * 4],
                                           initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable('b_hid', [n_h * 4], initializer=tf.zeros_initializer()))

            w_gate.append(tf.get_variable('w_gate', [n_h, k_gate],
                                           initializer=init_method(init_scale, init_mode)))
            b_gate.append(tf.get_variable('b_gate', [k_gate], initializer=tf.zeros_initializer()))

            tau_list.append(tf.get_variable('tau', [], initializer=tf.constant_initializer(1.0),
                                            trainable=True))


    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)  # N,n_h

    g_time_list = []    

    for t, (x_t, done_t) in enumerate(zip(xs, dones)):
        x_t   = tf.squeeze(x_t,   axis=0)  # N,n_s
        done_t= tf.squeeze(done_t,axis=0)  # scalar

        g_list = []
        m_list = []
        for i in range(n_agent):
            h_i = tf.expand_dims(h[i], axis=0)                   # 1,n_h
            g_logits = tf.matmul(h_i, w_gate[i]) + b_gate[i]     # 1,k
            g_i      = tf.nn.softmax(g_logits / tau_list[i])     # 1,k
            g_scalar = tf.reduce_max(g_i, axis=-1, keepdims=True)  # 1,1
            g_list.append(g_scalar)
            m_list.append(g_scalar * h_i)                      
        g_mat = tf.concat(g_list, axis=0)  # N,1
        m_mat = tf.concat(m_list, axis=0)  # N,n_h
        g_time_list.append(tf.expand_dims(g_mat, axis=1))      

        out_h, out_c = [], []
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            ci = ci * (1 - done_t)
            hi = hi * (1 - done_t)

            if use_attention:
                # Scaled-dot attention over neighbours
                nbr_idx   = np.where(masks[i])[0]
                h_nbr     = tf.gather(m_mat, nbr_idx)                    # n_n, n_h
                score     = tf.matmul(tf.matmul(hi, w_msg[i]), h_nbr, transpose_b=True) / tf.sqrt(float(n_h))
                attn_w    = tf.nn.softmax(score)                         # 1, n_n
                mi        = tf.matmul(attn_w, h_nbr)                     # 1, n_h
            else:
                mi_num = tf.reduce_sum(tf.boolean_mask(m_mat, masks[i]), axis=0, keepdims=True)
                gi_den = tf.reduce_sum(tf.boolean_mask(g_mat, masks[i]), axis=0, keepdims=True) + 1e-6
                mi      = mi_num / gi_den

            xi_nei = tf.expand_dims(tf.reshape(tf.boolean_mask(x_t, masks[i]), [-1]), axis=0)
            xi     = tf.concat([tf.expand_dims(x_t[i], axis=0), xi_nei], axis=1)
            s_i    = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i]) + tf.matmul(mi, w_msg[i]) + b_msg[i]

            if use_layer_norm:
                z_i = tf.matmul(s_i, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
                z_i = z_i = tf.keras.layers.LayerNormalization(axis=-1)(z_i)                  # LayerNorm
            else:
                z_i = tf.matmul(s_i, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]

            ii, fi, oi, ui = tf.split(z_i, 4, axis=1)
            ii = tf.sigmoid(ii); fi = tf.sigmoid(fi); oi = tf.sigmoid(oi); ui = tf.tanh(ui)
            ci = fi * ci + ii * ui
            if dropout_p > 0.0:
                ci = tf.nn.dropout(ci, rate=dropout_p)                    # recurrent dropout

            hi = oi * tf.tanh(ci)
            out_c.append(ci)
            out_h.append(hi)

        c = tf.concat(out_c, axis=0)
        h_prev = h
        h = tf.concat(out_h, axis=0)

        xs[t] = tf.expand_dims(h, axis=0) 
    new_s  = tf.concat([c, h], axis=1)          # N,2*n_h
    h_seq  = seq_to_batch(xs)                   # T,N,n_h
    h_seq  = tf.transpose(h_seq, [1,0,2])       # N,T,n_h
    g_seq  = tf.concat(g_time_list, axis=1)     # N,T,1

    return h_seq, new_s, g_seq


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma
        self.alpha = alpha
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=bool)
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            # additional spatial rewards
            for t in range(self.max_distance + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.alpha ** t) * rt
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        acts = np.transpose(np.array(self.acts, dtype=np.int32))
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=bool)
        self.reset(self.dones[-1])
        return obs, policies, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = r + self.gamma * cur_R * (1.-done)
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                # additional spatial rewards
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                # additional spatial rewards
                for t in range(max_distance + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)
        


"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


# BurstAttentionGAO

See:
https://arxiv.org/pdf/2403.09347.pdf

// Forward pass of GAO
void forward_gao(const Tensor& Q_i, const Tensor& K_i, const Tensor& V_i,
                 Tensor& O_i, Tensor& l_i, Tensor& m_i, int G) {
    int N = Q_i.size(0);
    int d = Q_i.size(1);
    
    O_i.resize_({N / G, d}).zero_();
    l_i.resize_({N / G}).zero_();
    m_i.resize_({N / G}).fill_(-INFINITY);
    
    Tensor K_buffer = K_i;
    Tensor V_buffer = V_i;
    
    for (int j = 0; j < G; j++) {
        // Conduct one step of ring communication
        Tensor K_j = K_buffer;
        Tensor V_j = V_buffer;
        
        // Forward pass of local attention
        Tensor S_ij = torch::matmul(Q_i, K_j.transpose(0, 1));
        Tensor m_ij = S_ij.max(1).values;
        Tensor P_ij = torch::exp(S_ij - m_ij.unsqueeze(1));
        Tensor l_ij = P_ij.sum(1);
        Tensor O_ij = torch::matmul(P_ij, V_j);
        
        // Aggregate local attention results
        Tensor m_new = torch::max(m_i, m_ij);
        l_i = torch::exp(m_i - m_new) * l_i + torch::exp(m_ij - m_new) * l_ij;
        O_i = torch::exp(m_i - m_new) * O_i + torch::exp(m_ij - m_new) * O_ij;
        m_i = m_new;
        
        // Put K_j, V_j into communication ring
        K_buffer = K_j;
        V_buffer = V_j;
    }
    
    O_i = O_i / l_i.unsqueeze(1);
    Tensor lse_i = m_i + torch::log(l_i);
    
    return {O_i, lse_i};
}

// Backward pass of GAO
std::tuple<Tensor, Tensor, Tensor> backward_gao(const Tensor& Q_i, const Tensor& K_i, const Tensor& V_i,
                                                const Tensor& O_i, const Tensor& dO_i, const Tensor& lse_i, int G) {
    int N = Q_i.size(0);
    int d = Q_i.size(1);
    
    Tensor dQ_i = torch::zeros_like(Q_i);
    Tensor dK_i = torch::zeros_like(K_i);
    Tensor dV_i = torch::zeros_like(V_i);
    
    Tensor D_i = (dO_i * O_i).sum(1);
    
    Tensor Q_buffer = Q_i;
    Tensor dQ_buffer = dQ_i;
    Tensor dO_buffer = dO_i;
    Tensor D_buffer = D_i;
    Tensor lse_buffer = lse_i;
    
    for (int j = 0; j < G; j++) {
        // Conduct one step of ring communication
        Tensor Q_j = Q_buffer;
        Tensor dQ_j = dQ_buffer;
        Tensor dO_j = dO_buffer;
        Tensor D_j = D_buffer;
        Tensor lse_j = lse_buffer;
        
        // Backward pass of local attention
        Tensor S_ji = torch::matmul(Q_j, K_i.transpose(0, 1));
        Tensor P_ji = torch::exp(S_ji - lse_j.unsqueeze(1));
        dV_i += torch::matmul(P_ji.transpose(0, 1), dO_j);
        Tensor dP_ji = torch::matmul(dO_j, V_i.transpose(0, 1));
        Tensor dS_ji = P_ji * (dP_ji - D_j.unsqueeze(1));
        dK_i += torch::matmul(dS_ji.transpose(0, 1), Q_j);
        dQ_j += torch::matmul(dS_ji, K_i);
        
        // Put Q_j, dQ_j, dO_j, D_j, lse_j into communication ring
        Q_buffer = Q_j;
        dQ_buffer = dQ_j;
        dO_buffer = dO_j;
        D_buffer = D_j;
        lse_buffer = lse_j;
    }
    
    return {dQ_i, dK_i, dV_i};
}

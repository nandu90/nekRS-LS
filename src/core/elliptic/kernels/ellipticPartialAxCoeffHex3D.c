#if p_knl == 0

#ifdef p_svv
#define DTERM(t,a) ((t) == 0 ? D[a] : Dsvv[a])
#define SR(t,a,b,c) ((t) == 0 ? s_Gqr[a][b][c] : s_Sqr[a][b][c])
#define SS(t,a,b,c) ((t) == 0 ? s_Gqs[a][b][c] : s_Sqs[a][b][c])
#define ST(t,a,b,c) ((t) == 0 ? s_Gqt[a][b][c] : s_Sqt[a][b][c])
#else
#define DTERM(t,a) D[a]
#define SR(t,a,b,c) s_Gqr[a][b][c]
#define SS(t,a,b,c) s_Gqs[a][b][c]
#define ST(t,a,b,c) s_Gqt[a][b][c]
#endif

extern "C" void FUNC(ellipticPartialAxCoeffHex3D_v0)(const dlong & Nelements,
                        const dlong & offset,
                        const dlong & loffset,
                        const dlong* __restrict__ elementList,
                        const gfloat* __restrict__ ggeo,
                        const dfloat* __restrict__ D,
                        const dfloat* __restrict__ S,
                        const dfloat* __restrict__ Dsvv,
                        const dfloat* __restrict__ lambda0,
                        const dfloat* __restrict__ lambda1,
                        const dfloat* __restrict__ lambdasvv,
                        const dfloat* __restrict__ q,
                        dfloat* __restrict__ Aq )
{
  dfloat s_q[p_Nq][p_Nq][p_Nq];
  dfloat s_Gqr[p_Nq][p_Nq][p_Nq];
  dfloat s_Gqs[p_Nq][p_Nq][p_Nq];
  dfloat s_Gqt[p_Nq][p_Nq][p_Nq];
#ifdef p_svv
  dfloat s_Sqr[p_Nq][p_Nq][p_Nq];
  dfloat s_Sqs[p_Nq][p_Nq][p_Nq];
  dfloat s_Sqt[p_Nq][p_Nq][p_Nq];
#endif

#ifdef __NEKRS__OMP__
#ifdef p_svv
  #pragma omp parallel for private(s_q, s_Gqr, s_Gqs, s_Gqt, s_Sqr, s_Sqs, s_Sqt)
#else
  #pragma omp parallel for private(s_q, s_Gqr, s_Gqs, s_Gqt)
#endif
#endif  
  for(dlong e = 0; e < Nelements; ++e) {
    const dlong element = elementList[e];

    for(int k = 0; k < p_Nq; k++)
      for(int j = 0; j < p_Nq; ++j)
        for(int i = 0; i < p_Nq; ++i) {
          const dlong base = i + j * p_Nq + k * p_Nq * p_Nq + element * p_Np;
          const dfloat qbase = q[base];
          s_q[k][j][i] = qbase;
        }

    for(int k = 0; k < p_Nq; ++k)
      for(int j = 0; j < p_Nq; ++j)
        for(int i = 0; i < p_Nq; ++i) {
          const dlong gbase = element * p_Nggeo * p_Np + k * p_Nq * p_Nq + j * p_Nq + i;
          const dfloat r_G00 = ggeo[gbase + p_G00ID * p_Np];
          const dfloat r_G01 = ggeo[gbase + p_G01ID * p_Np];
          const dfloat r_G11 = ggeo[gbase + p_G11ID * p_Np];
          const dfloat r_G12 = ggeo[gbase + p_G12ID * p_Np];
          const dfloat r_G02 = ggeo[gbase + p_G02ID * p_Np];
          const dfloat r_G22 = ggeo[gbase + p_G22ID * p_Np];

          const dlong id = element * p_Np + k * p_Nq * p_Nq + j * p_Nq + i;

          int dterms = 1;
#ifdef p_svv
          dterms++;
#endif
          const dfloat r_lam0 = lambda0[p_lambda*id + 0 * loffset];
          for (int t = 0; t < dterms; t++) {
            dfloat qr = 0;
            dfloat qs = 0;
            dfloat qt = 0;

            const dlong base = t == 0 ? 0 : element * p_Nq * p_Nq;
            for(int m = 0; m < p_Nq; m++){
              qr += DTERM(t, base + i*p_Nq + m) * s_q[k][j][m];
              qs += DTERM(t, base + j*p_Nq + m) * s_q[k][m][i];
              qt += DTERM(t, base + k*p_Nq + m) * s_q[m][j][i];
            }

            dfloat Gqr = r_G00 * qr;
            Gqr += r_G01 * qs;
            Gqr += r_G02 * qt;

            dfloat Gqs = r_G01 * qr;
            Gqs += r_G11 * qs;
            Gqs += r_G12 * qt;

            dfloat Gqt = r_G02 * qr;
            Gqt += r_G12 * qs;
            Gqt += r_G22 * qt;

            if (t == 0) {
              s_Gqr[k][j][i] = r_lam0 * Gqr;
              s_Gqs[k][j][i] = r_lam0 * Gqs;
              s_Gqt[k][j][i] = r_lam0 * Gqt;
            }
#ifdef p_svv
            else {
              s_Sqr[k][j][i] = Gqr;
              s_Sqs[k][j][i] = Gqs;
              s_Sqt[k][j][i] = Gqt;
            }
#endif          
          }
        }

    for(int k = 0; k < p_Nq; k++)
      for(int j = 0; j < p_Nq; ++j)
        for(int i = 0; i < p_Nq; ++i) {
          const dlong gbase = element * p_Nggeo * p_Np + k * p_Nq * p_Nq + j * p_Nq + i;

          const dlong id = element * p_Np + k * p_Nq * p_Nq + j * p_Nq + i;

          dfloat r_Aq = 0;
#ifndef p_poisson
          const dfloat r_lam1 = lambda1[p_lambda*id + 0 * loffset];
          r_Aq = ggeo[gbase + p_GWJID * p_Np] * r_lam1 * s_q[k][j][i];
#endif

          int dterms = 1;
#ifdef p_svv
          dterms++;
#endif          
          for(int t = 0; t < dterms; t++) {
            dfloat r_Aqr = 0, r_Aqs = 0, r_Aqt = 0;
            const dlong base = t == 0 ? 0 : element * p_Nq * p_Nq;

            for(int m = 0; m < p_Nq; m++){
              r_Aqr += DTERM(t, base + m*p_Nq +i) * SR(t,k,j,m);
              r_Aqs += DTERM(t, base + m*p_Nq +j) * SS(t,k,m,i);
              r_Aqt += DTERM(t, base + m*p_Nq +k) * ST(t,m,j,i);
            }

            const dfloat lamsvv = t == 0 ? 1.0 : lambdasvv[id];

            r_Aq += lamsvv * (r_Aqr + r_Aqs + r_Aqt);
          }

          Aq[id] = r_Aq;
        }
  }
}
#endif

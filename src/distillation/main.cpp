#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "distillation/nurbs_surface_wrapper.hpp"
#include "distillation/nurbs_mesh_processor.hpp"
#include "distillation/ruled_partitioner.hpp"
#include "distillation/boundary_smoother.hpp"
#include "distillation/occt_mesh_reconstruct.hpp"
#include "distillation/export_results.hpp"

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <vector>
#include <random>
#include <ctime>

using namespace distillation;

// ── Surfaces ────────────────────────────────────
static NurbsSurfaceWrapper createRandomSurface(int seed=0) {
    if(seed==0) seed=(int)std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(0.08,0.35);
    std::uniform_real_distribution<double> freq(1.2,4.5);
    std::uniform_real_distribution<double> phase(-1.0,1.0);

    int nU=9,nV=9,degU=3,degV=3;
    Vec3Arr cp(nU*nV);
    double uMin=-2.0,uMax=2.0,vMin=-2.0,vMax=2.0;

    // Generate 4-6 random wave components
    struct Wave { double ax,ay,fx,fy,px,py; };
    int nWaves=4+rng()%3;
    std::vector<Wave> waves(nWaves);
    for(auto& w:waves){
        w.ax=amp(rng); w.ay=amp(rng);
        w.fx=freq(rng); w.fy=freq(rng);
        w.px=phase(rng); w.py=phase(rng);
    }

    for(int i=0;i<nU;++i){
        double x=uMin+(uMax-uMin)*i/(nU-1);
        for(int j=0;j<nV;++j){
            double y=vMin+(vMax-vMin)*j/(nV-1);
            double z=0;
            for(auto& w:waves)
                z+=w.ax*std::sin(w.fx*x+w.px)*std::cos(w.fy*y+w.py)
                  +w.ay*std::cos(w.fx*x*0.7+w.px*1.3)*std::sin(w.fy*y*1.1+w.py*0.8);
            cp[i*nV+j]=Vec3(x,y,z);
        }
    }
    std::cout<<"  Random surface seed="<<seed<<" waves="<<nWaves<<"\n";
    return NurbsSurfaceWrapper(cp,nU,nV,
        makeClampedKnots(nU,degU,true),makeClampedKnots(nV,degV,true),degU,degV);
}
static NurbsSurfaceWrapper createWavySurface() {
    int nU=9,nV=9,degU=3,degV=3; Vec3Arr cp(nU*nV);
    std::vector<double> xs={-1.5,-1.125,-0.75,-0.375,0,0.375,0.75,1.125,1.5};
    std::vector<double> ys={-1.5,-1.125,-0.75,-0.375,0,0.375,0.75,1.125,1.5};
    for(int i=0;i<nU;++i)for(int j=0;j<nV;++j){
        double x=xs[i],y=ys[j];
        cp[i*nV+j]=Vec3(x,y,0.15*sin(2.5*x)*cos(3.0*y)+0.10*sin(5.0*x+1.2)*sin(4.0*y+0.8)+0.08*cos(7.0*x)*sin(6.0*y-0.5)+0.05*sin(9.0*x-1.0)*cos(8.0*y+1.5));
    }
    return NurbsSurfaceWrapper(cp,nU,nV,makeClampedKnots(nU,degU,true),makeClampedKnots(nV,degV,true),degU,degV);
}
static NurbsSurfaceWrapper createMountainTerrain() {
    int nU=12,nV=12,degU=3,degV=3; Vec3Arr cp(nU*nV);
    for(int i=0;i<nU;++i){double x=4.0*i/(nU-1),dx=x-2.0;
        for(int j=0;j<nV;++j){double y=4.0*j/(nV-1),dy=y-2.0;
            double ridge=std::max(1.2*exp(-pow((dx+dy)*0.5,2)*2.0),0.9*exp(-pow((dx-dy)*0.4,2)*3.0));
            double detail=0.15*sin(x*4.0)*cos(y*3.7)+0.10*sin(x*7.3+1.2)*sin(y*5.1+0.8)+0.06*cos(x*10.0+2.0)*sin(y*8.5+1.5)+0.04*sin(x*13.0*y*0.5);
            cp[i*nV+j]=Vec3(x,y,ridge+detail);
        }
    }
    return NurbsSurfaceWrapper(cp,nU,nV,makeClampedKnots(nU,degU,true),makeClampedKnots(nV,degV,true),degU,degV);
}

// ── Partition data export ───────────────────────
static void exportPartitionData(const std::string& dir,const IntArr& faceLabels,const FaceArr& faces,const Vec3Arr& verts3D,const Vec2Arr& uvs,int nParts){
    std::vector<IntArr> partFaces(nParts);
    for(int fi=0;fi<(int)faceLabels.size();++fi){int l=faceLabels[fi];if(l>=0&&l<nParts)partFaces[l].push_back(fi);}
    for(int pid=0;pid<nParts;++pid){
        if(partFaces[pid].empty())continue;
        std::set<int> pv;for(int fi:partFaces[pid]){pv.insert(faces[fi].v0);pv.insert(faces[fi].v1);pv.insert(faces[fi].v2);}
        std::map<std::pair<int,int>,int> ec;
        for(int fi:partFaces[pid]){const Face&f=faces[fi];for(int k=0;k<3;++k){int a=f[k],b=f[(k+1)%3];if(a>b)std::swap(a,b);ec[{a,b}]++;}}
        std::vector<std::pair<int,int>> be;for(auto&e:ec)if(e.second==1)be.push_back(e.first);
        if(be.size()<4)continue;
        std::map<int,IntArr> adj;for(auto&e:be){adj[e.first].push_back(e.second);adj[e.second].push_back(e.first);}
        std::vector<int> loop;int cur=adj.begin()->first,prev=-1;
        for(int s=0;s<1000;++s){loop.push_back(cur);auto&nb=adj[cur];int next=-1;for(int n:nb)if(n!=prev){next=n;break;}if(next<0||next==loop[0])break;prev=cur;cur=next;}
        if(loop.size()<4)continue;
        {std::ofstream out(dir+"/part_"+std::to_string(pid)+"_loop.txt");out.precision(10);for(int v:loop)out<<verts3D[v].x()<<" "<<verts3D[v].y()<<" "<<verts3D[v].z()<<"\n";}
        {std::ofstream out(dir+"/part_"+std::to_string(pid)+"_loop_uv.txt");out.precision(10);for(int v:loop)out<<uvs[v].x()<<" "<<uvs[v].y()<<"\n";}
        {std::ofstream out(dir+"/part_"+std::to_string(pid)+"_points.txt");out.precision(10);int maxN=200,step=std::max(1,(int)partFaces[pid].size()/maxN);
         for(int si=0;si<(int)partFaces[pid].size();si+=step){int fi=partFaces[pid][si];Vec3 c=(verts3D[faces[fi].v0]+verts3D[faces[fi].v1]+verts3D[faces[fi].v2])/3.0;out<<c.x()<<" "<<c.y()<<" "<<c.z()<<"\n";}}
    }
}

// ── main ─────────────────────────────────────────
int main(int argc,char*argv[]){
#ifdef _WIN32
    SetConsoleOutputCP(65001);SetConsoleCP(65001);
#endif
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered for GUI pipe
    std::string surface="random",exportDir="./results";
    double sigmaTarget=-1,tolTarget=-1;int smoothIters=-1,maxRetries=5;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a.find("--surface=")==0)surface=a.substr(10);
        else if(a.find("--export-dir=")==0){exportDir=a.substr(13); if(exportDir.size()>0&&exportDir[0]=='"')exportDir=exportDir.substr(1,exportDir.size()-2);}
        else if(a.find("--sigma=")==0)sigmaTarget=std::stod(a.substr(8));
        else if(a=="--smooth-iters"&&i+1<argc)smoothIters=std::stoi(argv[++i]);
        else if(a.find("--smooth-iters=")==0)smoothIters=std::stoi(a.substr(15));
        else if(a=="--tol-target"&&i+1<argc)tolTarget=std::stod(argv[++i]);
        else if(a.find("--tol-target=")==0)tolTarget=std::stod(a.substr(a.find('=')+1));
        else if(a=="--max-retries"&&i+1<argc)maxRetries=std::stoi(argv[++i]);
        else if(a.find("--max-retries=")==0)maxRetries=std::stoi(a.substr(a.find('=')+1));
        else if(a=="--help"||a=="-h"){std::cout<<"distillation [options]\n  --surface=<random|wavy|mountain>\n  --export-dir=<path>\n  --sigma=<val>\n  --smooth-iters=<n>\n  --tol-target=<val>\n  --max-retries=<n>\n";return 0;}
    }
    std::cout<<"=== NURBS Surface Partitioning + Boundary Smoothing ===\n";
    NurbsSurfaceWrapper nurbs;
    if(surface=="mountain")nurbs=createMountainTerrain();
    else if(surface=="wavy")nurbs=createWavySurface();
    else nurbs=createRandomSurface();
    std::cout<<"  Surface: "<<nurbs.numCtrlU()<<"x"<<nurbs.numCtrlV()<<" ctrl, deg "<<nurbs.degreeU()<<"x"<<nurbs.degreeV()<<"\n";
    Vec3Arr vertices;FaceArr faces;Vec2Arr uvs;nurbs.generateMesh(60,60,vertices,faces,uvs);
    std::cout<<"  Mesh: "<<vertices.size()<<" verts, "<<faces.size()<<" faces\n";

    double sigma=(sigmaTarget>0)?sigmaTarget:0;
    int K_parts=16, prevNFail=999;

    for(int retry=0;retry<maxRetries;++retry){
        std::string retryDir = exportDir + "/retry_" + std::to_string(retry);
        std::filesystem::create_directories(retryDir);
        if(retry>0)std::cout<<"\n  [Retry "<<retry<<"] K="<<K_parts<<" sigma="<<sigma<<"\n";

        // ── Hard-EM with current K_parts ──
        HardEMPartitioner partitioner(nurbs,K_parts,6,3,0.001,20,0.0,true);
        auto partResult=partitioner.partition(vertices,uvs);
        const IntVecSet& partitions=std::get<0>(partResult);
        int nonEmpty=0;for(auto&p:partitions)if(!p.empty())nonEmpty++;
        std::cout<<"  Partitions: "<<nonEmpty<<"/"<<K_parts<<"\n";

        IntArr faceLabels=convertVertexLabelsToFaceLabels(partitions,faces,(int)vertices.size());
        mergeTinyRegions(faceLabels,faces);

        auto[uMin,uMax]=nurbs.paramDomainU();auto[vMin,vMax]=nurbs.paramDomainV();
        BoundaryNetwork net=extractBoundaryNetwork(uvs,faces,faceLabels,uMin,uMax,vMin,vMax);
        std::cout<<"  Boundary: "<<net.localToGlobal.size()<<" verts, "<<net.edges.size()<<" edges\n";
        if(net.edges.empty()){exportOBJ(retryDir+"/mesh.obj",vertices,faces);return 0;}

        if(sigma<=0)sigma=net.avgEdgeLength*2.0;

        int K=1;
        if(retry==0&&smoothIters>0)K=smoothIters;
        else if(net.avgEdgeLength>EPS)K=(int)std::ceil(pow(2.0*sigma/net.avgEdgeLength,2.0));
        K=std::max(1,std::min(K,200));
        std::cout<<"  [Laplacian] sigma="<<sigma<<" K="<<K<<"\n";

        if(retry>0){net.smoothedUVs.clear();net.smoothedUVs.resize(net.localToGlobal.size());
            for(int i=0;i<(int)net.localToGlobal.size();++i)net.smoothedUVs[i]=uvs[net.localToGlobal[i]];}

        {auto p0=extractPolylinesFromNetwork(net);exportBoundaryPolylinesIter(retryDir+"/boundaries_iter_000.txt",p0,nurbs,0);}
        std::vector<SmoothIterationEntry> smoothHistory;
        for(int iter=1;iter<=K;++iter){
            auto entry=laplacianSmoothSingle(net,uMin,uMax,vMin,vMax);entry.iteration=iter;smoothHistory.push_back(entry);
            if(iter%std::max(1,K/10)==0||iter==K)std::cout<<"    iter "<<iter<<"/"<<K<<" maxDisp="<<entry.maxDisplacement<<"\n";
            auto pi=extractPolylinesFromNetwork(net);std::ostringstream oss;oss<<retryDir<<"/boundaries_iter_"<<std::setw(3)<<std::setfill('0')<<iter<<".txt";
            exportBoundaryPolylinesIter(oss.str(),pi,nurbs,iter);
            if(entry.maxDisplacement<1e-6)break;
        }

        Vec2Arr updatedUVs=uvs;
        for(int i=0;i<(int)net.localToGlobal.size();++i)updatedUVs[net.localToGlobal[i]]=net.smoothedUVs[i];
        harmonicMeshUpdate(updatedUVs,faces,net);
        Vec3Arr updatedVerts3D=liftMeshTo3D(updatedUVs,nurbs);
        auto polylines2D=extractPolylinesFromNetwork(net);

        std::cout<<"  [Step 3] OCCT Splitter...\n";
        auto[reconUVs,reconFaces,reconFaceLabels]=occtConstrainedReconstruct(uvs,faces,faceLabels,net,uMin,uMax,vMin,vMax);

        std::vector<Vec3Arr> boundaryCurves3D;
        for(auto&poly:polylines2D){Vec3Arr pts;for(auto&uv:poly)pts.push_back(nurbs.evaluate(uv.x(),uv.y()));if(!pts.empty())boundaryCurves3D.push_back(pts);}

        exportPartitionData(retryDir,faceLabels,faces,updatedVerts3D,updatedUVs,(int)partitions.size());

        Vec3Arr corners3D;
        for(auto&poly:polylines2D){if(poly.size()<8)continue;
            for(size_t i=0;i<poly.size();++i){const Vec2&p0=poly[(i+poly.size()-1)%poly.size()],&p1=poly[i],&p2=poly[(i+1)%poly.size()];
                Vec2 v1=p1-p0,v2=p2-p1;double n1=v1.norm(),n2=v2.norm();if(n1<1e-8||n2<1e-8)continue;
                double cosA=clamp(v1.normalized().dot(v2.normalized()),-1.,1.);if(radToDeg(std::acos(cosA))>60.)corners3D.push_back(nurbs.evaluate(p1.x(),p1.y()));}}

        exportOBJ(retryDir+"/mesh.obj",updatedVerts3D,faces);
        {IntArr vl(updatedVerts3D.size(),-1);for(int fi=0;fi<(int)faceLabels.size();++fi){int l=faceLabels[fi];if(l>=0){vl[faces[fi].v0]=l;vl[faces[fi].v1]=l;vl[faces[fi].v2]=l;}}exportPartitionLabels(retryDir+"/partition_labels.txt",(int)updatedVerts3D.size(),vl);}
        exportFaceLabels(retryDir+"/face_labels.txt",faceLabels);
        exportOBJ(retryDir+"/mesh_original.obj",vertices,faces);
        {Vec3Arr rv=liftMeshTo3D(reconUVs,nurbs);exportOBJ(retryDir+"/mesh_recon.obj",rv,reconFaces);exportFaceLabels(retryDir+"/face_labels_recon.txt",reconFaceLabels);}
        exportBoundaryCurves3D(retryDir+"/boundaries.txt",boundaryCurves3D);
        exportBoundaryCurves2D(retryDir+"/boundaries_uv.txt",polylines2D);
        if(!corners3D.empty())exportCorners(retryDir+"/corners.txt",corners3D);
        if(!smoothHistory.empty())exportSmoothHistory(retryDir+"/smooth_history.txt",smoothHistory);
        {std::ofstream meta(retryDir+"/run_meta.txt");meta<<"surface="<<surface
          <<"\nK="<<K<<"\nsigma="<<sigma<<"\nactual_iters="<<smoothHistory.size()
          <<"\nnurbs_ctrl_u="<<nurbs.numCtrlU()<<"\nnurbs_ctrl_v="<<nurbs.numCtrlV()
          <<"\nnurbs_degree_u="<<nurbs.degreeU()<<"\nnurbs_degree_v="<<nurbs.degreeV()
          <<"\nnurbs_domain_u="<<uMin<<" "<<uMax
          <<"\nnurbs_domain_v="<<vMin<<" "<<vMax
          <<"\n";}
        // Export NURBS surface definition for Python
        {std::ofstream ns(retryDir+"/nurbs_surface.txt");
         Handle(Geom_BSplineSurface) s=nurbs.surface();
         ns.precision(12);
         ns<<nurbs.numCtrlU()<<" "<<nurbs.numCtrlV()<<" "
           <<nurbs.degreeU()<<" "<<nurbs.degreeV()<<"\n";
         // Expand compact knots to full form (with multiplicities)
         ns<<(nurbs.numCtrlU()+nurbs.degreeU()+1)<<" ";
         auto& ku=s->UKnots(); auto& mu=s->UMultiplicities();
         for(int i=1;i<=ku.Length();++i)
             for(int m=0;m<mu.Value(i);++m)ns<<ku.Value(i)<<" ";
         ns<<"\n";
         ns<<(nurbs.numCtrlV()+nurbs.degreeV()+1)<<" ";
         auto& kv=s->VKnots(); auto& mv=s->VMultiplicities();
         for(int i=1;i<=kv.Length();++i)
             for(int m=0;m<mv.Value(i);++m)ns<<kv.Value(i)<<" ";
         ns<<"\n";
         for(int i=0;i<nurbs.numCtrlU();++i)
           for(int j=0;j<nurbs.numCtrlV();++j){
             gp_Pnt p=s->Pole(i+1,j+1); double w=s->Weight(i+1,j+1);
             ns<<p.X()<<" "<<p.Y()<<" "<<p.Z()<<" "<<w<<"\n";}
        }

        std::cout<<"\n=== Done ===\n  Partition data: "<<retryDir<<"\n  Optimizer: python python\\fit_ruled_grad.py "<<retryDir<<"\n";

        if(tolTarget>0){
            std::string cmd="python -u python\\fit_ruled_grad.py "+retryDir+" --max-iter 3 --lr 0.02";
            std::cout<<"  [Tol] running: "<<cmd<<"\n"<<std::flush;
            FILE* pipe = _popen(cmd.c_str(), "r");
            if(pipe){
                char buf[1024];
                while(fgets(buf,sizeof(buf),pipe)){std::cout<<buf<<std::flush;}
                _pclose(pipe);
            }
            std::ifstream tfin(retryDir+"/tolerance.txt");
            std::vector<double> partTols;
            if(tfin){std::string line;std::getline(tfin,line);
                while(std::getline(tfin,line)){std::istringstream iss(line);int pid;double tp,tq,md,rd;
                    if(iss>>pid>>tp>>tq>>md>>rd){partTols.resize(std::max((int)partTols.size(),pid+1),0.);partTols[pid]=md;}}}
            int nFail=0;for(size_t pi=0;pi<partTols.size();++pi)if(partTols[pi]>tolTarget)++nFail;
            if(nFail>0&&retry+1<maxRetries){
                if(nFail>=prevNFail){K_parts+=2;std::cout<<"  [Tol] "<<nFail<<"/"<<partTols.size()<<" exceed "<<tolTarget<<", K->"<<K_parts<<"\n";}
                else{sigma*=0.7;std::cout<<"  [Tol] "<<nFail<<"/"<<partTols.size()<<" exceed "<<tolTarget<<", sigma->"<<sigma<<"\n";}
                prevNFail=nFail;continue;
            }else if(nFail>0){
                std::cout<<"  [Tol] "<<nFail<<" still exceed after retries, stop\n";
            }else{
                std::cout<<"  [Tol] all "<<partTols.size()<<" within "<<tolTarget<<", done\n";
            }
        }
        break;
    }
    return 0;
}

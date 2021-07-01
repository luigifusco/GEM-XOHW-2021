# /******************************************
# *MIT License
# *
# *Copyright (c} [2021] [Luigi Fusco, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"}, to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# */

#checkenv
echo ""
echo "***************************************"
echo "[GEM-Info] check Vivado"
echo ""
echo "***************************************"
if [[ -z "$XILINX_VIVADO" ]]; then
    echo "Must source Vivado (hence defining XILINX_VIVADO} in environment" 1>&2
    exit 1
fi
TOP=$(pwd)


#setup
echo "[GEM-Info] set up the environment"
BRD_PARTS="xczu7ev-ffvc1156-2-e"
CLK_NAME=fclk0_mhz
HLS_CLK=10
FREQ_MHZ=100



KERNEL=gem
BOARD_BITSTREAM=${KERNEL}_wrapper.bit
CURR_BUILD_DIR=${TOP}/build
VIVADO_SCRIPT_DIR=${TOP}/scripts
SCRIPT_DIR=${TOP}/scripts

VIVADO_PRJNAME=${KERNEL}-vivado
HLS_PRJNAME=${KERNEL}-hls

BITSTREAM=${PRJDIR}/${VIVADO_PRJNAME}.runs/impl_1/${KERNEL}_wrapper.bit
VVD_SCRIPT=${VIVADO_SCRIPT_DIR}/create_vivado_project.tcl
VVD_SYNTH_SCRIPT=${VIVADO_SCRIPT_DIR}/synth_vivado_project.tcl



# begin


echo ""
echo "***************************************"
echo "[GEM-Info] asset folder creation"
echo ""
echo "***************************************"

mkdir -p ${CURR_BUILD_DIR}/assets


########build 1
echo ""
echo "***************************************"
echo "[GEM] Build the MI kernel"
echo ""
echo "***************************************"

TRGT_CORE=mutual_information
mkdir -p ${CURR_BUILD_DIR}/${TRGT_CORE}
PRJDIR=${CURR_BUILD_DIR}/${TRGT_CORE}/${VIVADO_PRJNAME}
SRC_DIR=${TOP}/metrics/${TRGT_CORE}
HLS_CODE=($(ls ${SRC_DIR}/parzen.cpp))
HLS_CODE+=($(ls ${SRC_DIR}/*.h))
HLS_CODE+=($(ls ${SRC_DIR}/*.hpp))
HLS_CODE_STRING="${HLS_CODE[@]}"
IP_REPO=${CURR_BUILD_DIR}/${TRGT_CORE}/${HLS_PRJNAME}/solution1/impl/ip

echo "${HLS_CODE[@]}"
echo "$HLS_CODE_STRING"

CORE_NAME=parzen_master
BITSTREAM=${PRJDIR}/${VIVADO_PRJNAME}.runs/impl_1/${KERNEL}_wrapper.bit

#HLS
echo ""
echo "***************************************"
echo "[GEM-Info] Starting HLS for MI kernel"
echo ""
echo "***************************************"
cd ${CURR_BUILD_DIR}/${TRGT_CORE}
vivado_hls -f ${SCRIPT_DIR}/hls.tcl -tclargs ${HLS_PRJNAME} "${HLS_CODE_STRING}" ${BRD_PARTS} ${HLS_CLK} ${CORE_NAME} "${SRC_DIR}/";
cd ../

#vivado
echo ""
echo "***************************************"
echo "[GEM-Info] Starting Vivado for MI kernel"
echo ""
echo "***************************************"
vivado -mode batch -source ${VVD_SCRIPT} -tclargs ${TOP} ${VIVADO_PRJNAME} ${PRJDIR} ${IP_REPO} ${FREQ_MHZ} 1 ${CORE_NAME} ${KERNEL} 0
vivado -mode batch -source ${VVD_SYNTH_SCRIPT} -tclargs ${PRJDIR}/${VIVADO_PRJNAME}.xpr ${PRJDIR} ${VIVADO_PRJNAME} ${KERNEL}_wrapper
mkdir -p ${CURR_BUILD_DIR}/assets/${TRGT_CORE}
cp ${BITSTREAM} ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.bit
cp ${PRJDIR}/${KERNEL}_wrapper.tcl ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/
cp ${PRJDIR}/${VIVADO_PRJNAME}.srcs/sources_1/bd/${KERNEL}/hw_handoff/${KERNEL}.hwh ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.hwh




########build 2
echo ""
echo "***************************************"
echo "[GEM] Build the Grandient Matrix MI kernel"
echo ""
echo "***************************************"

TRGT_CORE=mutual_information_gradient_matrix
mkdir -p ${CURR_BUILD_DIR}/${TRGT_CORE}
PRJDIR=${CURR_BUILD_DIR}/${TRGT_CORE}/${VIVADO_PRJNAME}
SRC_DIR=${TOP}/metrics/${TRGT_CORE}
HLS_CODE=($(ls ${SRC_DIR}/mutual_information_derived.cpp))
HLS_CODE+=($(ls ${SRC_DIR}/*.h))
HLS_CODE+=($(ls ${SRC_DIR}/*.hpp))
HLS_CODE_STRING="${HLS_CODE[@]}"
IP_REPO=${CURR_BUILD_DIR}/${TRGT_CORE}/${HLS_PRJNAME}/solution1/impl/ip

echo "${HLS_CODE[@]}"
echo "$HLS_CODE_STRING"

CORE_NAME=mutual_information_derived_master
BITSTREAM=${PRJDIR}/${VIVADO_PRJNAME}.runs/impl_1/${KERNEL}_wrapper.bit

#HLS
echo ""
echo "***************************************"
echo "[GEM-Info] Starting HLS for Grandient MI Matrix kernel"
echo ""
echo "***************************************"
cd ${CURR_BUILD_DIR}/${TRGT_CORE}
vivado_hls -f ${SCRIPT_DIR}/hls.tcl -tclargs ${HLS_PRJNAME} "${HLS_CODE_STRING}" ${BRD_PARTS} ${HLS_CLK} ${CORE_NAME} "${SRC_DIR}/";
cd ../

#vivado
echo ""
echo "***************************************"
echo "[GEM-Info] Starting Vivado for Grandient MI Matrix kernel"
echo ""
echo "***************************************"
vivado -mode batch -source ${VVD_SCRIPT} -tclargs ${TOP} ${VIVADO_PRJNAME} ${PRJDIR} ${IP_REPO} ${FREQ_MHZ} 1 ${CORE_NAME} ${KERNEL} 0
vivado -mode batch -source ${VVD_SYNTH_SCRIPT} -tclargs ${PRJDIR}/${VIVADO_PRJNAME}.xpr ${PRJDIR} ${VIVADO_PRJNAME} ${KERNEL}_wrapper
mkdir -p ${CURR_BUILD_DIR}/assets/${TRGT_CORE}
cp ${BITSTREAM} ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.bit
cp ${PRJDIR}/${KERNEL}_wrapper.tcl ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/
cp ${PRJDIR}/${VIVADO_PRJNAME}.srcs/sources_1/bd/${KERNEL}/hw_handoff/${KERNEL}.hwh ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.hwh

########build 3
echo ""
echo "***************************************"
echo "[GEM] Build the Grandient MI kernel"
echo ""
echo "***************************************"

TRGT_CORE=mutual_information_gradient
mkdir -p ${CURR_BUILD_DIR}/${TRGT_CORE}
PRJDIR=${CURR_BUILD_DIR}/${TRGT_CORE}/${VIVADO_PRJNAME}
SRC_DIR=${TOP}/metrics/${TRGT_CORE}
HLS_CODE=($(ls ${SRC_DIR}/mutual_information_derived.cpp))
HLS_CODE+=($(ls ${SRC_DIR}/*.h))
HLS_CODE+=($(ls ${SRC_DIR}/*.hpp))
HLS_CODE_STRING="${HLS_CODE[@]}"
IP_REPO=${CURR_BUILD_DIR}/${TRGT_CORE}/${HLS_PRJNAME}/solution1/impl/ip

echo "${HLS_CODE[@]}"
echo "$HLS_CODE_STRING"

CORE_NAME=mutual_information_derived_master
BITSTREAM=${PRJDIR}/${VIVADO_PRJNAME}.runs/impl_1/${KERNEL}_wrapper.bit

#HLS
echo ""
echo "***************************************"
echo "[GEM-Info] Starting HLS for Grandient MI kernel"
echo ""
echo "***************************************"
cd ${CURR_BUILD_DIR}/${TRGT_CORE}
vivado_hls -f ${SCRIPT_DIR}/hls.tcl -tclargs ${HLS_PRJNAME} "${HLS_CODE_STRING}" ${BRD_PARTS} ${HLS_CLK} ${CORE_NAME} "${SRC_DIR}/";
cd ../

#vivado
echo ""
echo "***************************************"
echo "[GEM-Info] Starting Vivado for Grandient MI kernel"
echo ""
echo "***************************************"
vivado -mode batch -source ${VVD_SCRIPT} -tclargs ${TOP} ${VIVADO_PRJNAME} ${PRJDIR} ${IP_REPO} ${FREQ_MHZ} 1 ${CORE_NAME} ${KERNEL} 1
vivado -mode batch -source ${VVD_SYNTH_SCRIPT} -tclargs ${PRJDIR}/${VIVADO_PRJNAME}.xpr ${PRJDIR} ${VIVADO_PRJNAME} ${KERNEL}_wrapper
mkdir -p ${CURR_BUILD_DIR}/assets/${TRGT_CORE}
cp ${BITSTREAM} ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.bit
cp ${PRJDIR}/${KERNEL}_wrapper.tcl ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/
cp ${PRJDIR}/${VIVADO_PRJNAME}.srcs/sources_1/bd/${KERNEL}/hw_handoff/${KERNEL}.hwh ${CURR_BUILD_DIR}/assets/${TRGT_CORE}/${KERNEL}_wrapper.hwh

echo ""
echo "***************************************"
echo "[GEM] GEM is at the end, bye"
echo ""
echo "***************************************"

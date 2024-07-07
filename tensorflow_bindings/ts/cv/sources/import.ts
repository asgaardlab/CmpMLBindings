import {LayersModel, loadLayersModel} from './tfimport';

export const importModel = async (path: string): Promise<LayersModel> =>
    loadLayersModel('file://' + path);

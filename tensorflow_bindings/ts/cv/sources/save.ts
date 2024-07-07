import {Sequential, SaveResult} from './tfimport';

export const saveModel = async (
    model: Sequential,
    path: string
): Promise<SaveResult> => model.save(path);
